"""Tests the branching logic of run_tick's helper functions with everything
external mocked (no live MLflow, no live Postgres, no Evidently install in
this sandbox). Verifies decision wiring: gate skipped when retrain isn't
triggered, promote called only when the gate says promote, rejected
challengers still get an audit_log entry.
"""

import tests._stubs  # noqa: F401  (must run before src imports below)

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.gate.evaluate import GateResult
from src.model.features import TARGET
from src.orchestration.pipeline import retrain_and_gate


def make_labeled_batch(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            TARGET: rng.choice([0, 1], size=n, p=[0.9, 0.1]),
            "RevolvingUtilizationOfUnsecuredLines": rng.random(n),
            "age": rng.integers(21, 90, n),
            "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 3, n),
            "DebtRatio": rng.random(n),
            "MonthlyIncome": rng.uniform(1000, 8000, n),
            "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 20, n),
            "NumberOfTimes90DaysLate": rng.integers(0, 3, n),
            "NumberRealEstateLoansOrLines": rng.integers(0, 5, n),
            "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 3, n),
            "NumberOfDependents": rng.integers(0, 4, n),
        }
    )


GATE_CONFIG = {
    "primary_metric": "auc_pr",
    "decision_threshold": 0.5,
    "tolerance_band": 0.01,
    "significance_alpha": 0.05,
    "mcnemar_min_discordant_pairs": 15,
    "bootstrap_resamples": 500,
}


def test_retrain_and_gate_writes_audit_log_on_rejection():
    """A rejected challenger must still produce an audit_log entry - this is
    the fix from the review: rejections are events, not silence.
    """
    batch = make_labeled_batch()
    conn = MagicMock()

    with patch("src.orchestration.pipeline.train_challenger") as mock_train, \
         patch("src.orchestration.pipeline.score_model") as mock_score, \
         patch("src.orchestration.pipeline.write_audit_log") as mock_audit:
        mock_train.return_value = ("run_1", "v2", MagicMock())
        # champion and challenger score identically -> dominance fails -> reject
        mock_score.side_effect = lambda model, df: np.full(len(df), 0.5)

        outcome = retrain_and_gate(
            conn, batch, production_model=MagicMock(),
            training_pool_df=batch, gate_config=GATE_CONFIG, run_name="test",
        )

    assert outcome["gate_result"].promote is False
    mock_audit.assert_called_once()
    call_kwargs = mock_audit.call_args
    assert call_kwargs.kwargs["event_type"] == "gate_evaluation"
    assert call_kwargs.kwargs["payload"]["promote"] is False


def test_retrain_and_gate_reports_promotion_when_challenger_clearly_wins():
    batch = make_labeled_batch(n=400, seed=1)
    conn = MagicMock()

    with patch("src.orchestration.pipeline.train_challenger") as mock_train, \
         patch("src.orchestration.pipeline.score_model") as mock_score, \
         patch("src.orchestration.pipeline.write_audit_log") as mock_audit:
        mock_train.return_value = ("run_2", "v3", MagicMock())

        def fake_score(model, df):
            y = df[TARGET].to_numpy()
            if model == "champion_sentinel":
                return np.where(y == 1, 0.4, 0.4)
            return np.where(y == 1, 0.9, 0.1)

        mock_score.side_effect = fake_score

        outcome = retrain_and_gate(
            conn, batch, production_model="champion_sentinel",
            training_pool_df=batch, gate_config=GATE_CONFIG, run_name="test",
        )

    assert outcome["gate_result"].promote is True
    mock_audit.assert_called_once()
    assert mock_audit.call_args.kwargs["payload"]["promote"] is True
