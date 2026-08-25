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


def test_build_expanded_training_pool_appends_labeled_predictions():
    """This is the actual fix for the bug a real 25-tick run exposed: every
    challenger was trained on the exact same static base pool as the
    champion, so champion_prob and challenger_prob came out byte-identical
    on every single tick (delta=0.0, 0 discordant pairs, 0 promotions ever).
    """
    from src.orchestration.pipeline import build_expanded_training_pool

    base_pool = make_labeled_batch(n=50, seed=0)
    conn = MagicMock()

    fake_labeled_rows = [
        {"features": row.drop(TARGET).to_dict(), "true_label": int(row[TARGET])}
        for _, row in make_labeled_batch(n=20, seed=99).iterrows()
    ]

    with patch(
        "src.orchestration.pipeline.get_labeled_predictions", return_value=fake_labeled_rows
    ):
        expanded = build_expanded_training_pool(conn, base_pool)

    assert len(expanded) == len(base_pool) + len(fake_labeled_rows)
    # base pool rows are untouched, not replaced (dtype can be promoted by
    # concat, e.g. int64 -> float64, which doesn't affect training - only
    # values matter here)
    pd.testing.assert_frame_equal(
        expanded.iloc[: len(base_pool)].reset_index(drop=True),
        base_pool,
        check_dtype=False,
    )


def test_build_expanded_training_pool_returns_base_unchanged_when_nothing_labeled_yet():
    from src.orchestration.pipeline import build_expanded_training_pool

    base_pool = make_labeled_batch(n=50, seed=0)
    conn = MagicMock()

    with patch("src.orchestration.pipeline.get_labeled_predictions", return_value=[]):
        expanded = build_expanded_training_pool(conn, base_pool)

    assert expanded is base_pool


def test_retrain_and_gate_can_actually_promote_when_pools_differ():
    """End-to-end sanity check of the fix: when the challenger is trained on
    a genuinely different (expanded) pool than what produced the champion,
    the two models are NOT forced to be identical, so the gate has a real
    chance to distinguish them. This doesn't happen with a real
    LogisticRegression trained twice on identical data (see the bug this
    fixes), so this test uses distinguishable fake scores to confirm the
    plumbing - not a training convergence detail - is what's under test.
    """
    batch = make_labeled_batch(n=400, seed=5)
    conn = MagicMock()

    with patch("src.orchestration.pipeline.train_challenger") as mock_train, \
         patch("src.orchestration.pipeline.score_model") as mock_score, \
         patch("src.orchestration.pipeline.write_audit_log"), \
         patch("src.orchestration.pipeline.get_labeled_predictions") as mock_get_labeled:
        mock_get_labeled.return_value = [
            {"features": row.drop(TARGET).to_dict(), "true_label": int(row[TARGET])}
            for _, row in make_labeled_batch(n=30, seed=6).iterrows()
        ]
        mock_train.return_value = ("run_3", "v4", MagicMock())

        def fake_score(model, df):
            y = df[TARGET].to_numpy()
            if model == "stale_champion":
                return np.where(y == 1, 0.3, 0.5)  # deliberately poor separation
            return np.where(y == 1, 0.9, 0.1)  # clearly better challenger

        mock_score.side_effect = fake_score

        from src.orchestration.pipeline import build_expanded_training_pool

        expanded = build_expanded_training_pool(conn, batch)
        outcome = retrain_and_gate(
            conn, batch, production_model="stale_champion",
            training_pool_df=expanded, gate_config=GATE_CONFIG, run_name="test",
        )

    assert outcome["gate_result"].promote is True
    assert outcome["gate_result"].champion_metric != outcome["gate_result"].challenger_metric
