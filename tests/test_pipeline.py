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
from src.orchestration.pipeline import retrain_and_gate, run_tick


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


def test_build_expanded_training_pool_caps_large_base_pool_so_recent_labels_carry_weight():
    """This is the second fix a real 25-tick run exposed: even after fixing
    the identical-predictions bug, a full ~127K-row base pool drowned out a
    few hundred newly-labeled post-drift rows into statistical noise (never
    more than a handful of discordant pairs, bootstrap CIs always straddling
    zero) - 0 promotions across a full run despite real, persistent drift.
    Capping the base pool's contribution lets accumulated labels actually
    move the needle.
    """
    from src.orchestration.pipeline import MAX_BASE_POOL_SAMPLE, build_expanded_training_pool

    large_base_pool = make_labeled_batch(n=MAX_BASE_POOL_SAMPLE + 5000, seed=0)
    conn = MagicMock()
    fake_labeled_rows = [
        {"features": row.drop(TARGET).to_dict(), "true_label": int(row[TARGET])}
        for _, row in make_labeled_batch(n=100, seed=99).iterrows()
    ]

    with patch(
        "src.orchestration.pipeline.get_labeled_predictions", return_value=fake_labeled_rows
    ):
        expanded = build_expanded_training_pool(conn, large_base_pool)

    # base pool contribution is capped, not the full ~132K rows
    assert len(expanded) == MAX_BASE_POOL_SAMPLE + len(fake_labeled_rows)
    # recent labeled data now makes up a meaningful share, not a rounding error
    recent_share = len(fake_labeled_rows) / len(expanded)
    assert recent_share > 0.01  # would be ~0.00075 without the cap on a 132K pool


def test_build_expanded_training_pool_does_not_cap_when_base_pool_already_small():
    from src.orchestration.pipeline import build_expanded_training_pool

    small_base_pool = make_labeled_batch(n=50, seed=0)
    conn = MagicMock()
    fake_labeled_rows = [
        {"features": row.drop(TARGET).to_dict(), "true_label": int(row[TARGET])}
        for _, row in make_labeled_batch(n=10, seed=99).iterrows()
    ]

    with patch(
        "src.orchestration.pipeline.get_labeled_predictions", return_value=fake_labeled_rows
    ):
        expanded = build_expanded_training_pool(conn, small_base_pool)

    # base pool is well under the cap, so all of it is kept, none dropped
    assert len(expanded) == len(small_base_pool) + len(fake_labeled_rows)


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


def test_run_tick_skips_rollback_check_on_the_tick_it_promotes():
    """Fix: right after a promotion, the newly-promoted challenger and the
    window it was gated against are the same model scored on the same rows
    check_rollback stored window_metrics from. Running the ordinary
    rollback comparison there is a no-op dressed as a real check (~zero
    drop, fresh fingerprint, by construction). run_tick must skip it and
    write an explicit skipped rollback_check audit entry instead of calling
    check_rollback at all on that tick.
    """
    batch_df = make_labeled_batch(n=400, seed=10)
    holdout_df = make_labeled_batch(n=400, seed=11)
    training_pool_df = make_labeled_batch(n=100, seed=12)
    conn = MagicMock()

    def fake_score(model, df):
        y = df[TARGET].to_numpy()
        if model == "champion_model":
            return np.where(y == 1, 0.4, 0.4)  # champion: no separation
        return np.where(y == 1, 0.9, 0.1)  # challenger: clear separation

    config = {
        "delayed_labels": {"delay_batches": 0},
        "gate": {
            **GATE_CONFIG,
            "retrain_drift_share_threshold": 0.3,
            "rollback_metric_drop_threshold": 0.03,
            "staleness_drift_share_delta_threshold": 0.3,
            "staleness_column_pvalue_delta_threshold": 0.3,
        },
    }

    audit_calls = []

    def fake_write_audit_log(conn, event_type, payload):
        audit_calls.append({"event_type": event_type, "payload": payload})

    with patch("src.orchestration.pipeline._production_cache") as mock_cache, \
         patch("src.orchestration.pipeline.check_retrain_trigger") as mock_trigger, \
         patch("src.orchestration.pipeline.release_due_labels"), \
         patch(
             "src.orchestration.pipeline.build_expanded_training_pool",
             return_value=training_pool_df,
         ), \
         patch("src.orchestration.pipeline.train_challenger") as mock_train, \
         patch("src.orchestration.pipeline.score_model", side_effect=fake_score), \
         patch("src.orchestration.pipeline.promote_challenger", return_value=777) as mock_promote, \
         patch("src.orchestration.pipeline.check_rollback") as mock_rollback, \
         patch("src.orchestration.pipeline.write_audit_log", side_effect=fake_write_audit_log), \
         patch("src.orchestration.pipeline.inject_drift", side_effect=lambda df, idx, cfg: df), \
         patch("src.orchestration.pipeline.insert_predictions_bulk"):
        mock_cache.get.return_value = ("champion_model", "v1")
        mock_trigger.return_value = (True, {"drift_share": 0.5})
        mock_train.return_value = ("run_x", "v2", MagicMock())

        result = run_tick(
            conn,
            current_batch=0,
            raw_batches=[batch_df],
            training_pool_df=training_pool_df,
            config=config,
            holdout_df=holdout_df,
        )

    assert result["promoted_champion_history_id"] == 777
    mock_promote.assert_called_once()

    # the real rollback check must never run on this tick
    mock_rollback.assert_not_called()

    # an explicit, honestly-labeled skip must be recorded instead
    assert result["rollback"]["skipped"] is True
    assert result["rollback"]["rollback_triggered"] is False
    assert result["rollback"]["rollback_executed"] is False

    rollback_audit_entries = [c for c in audit_calls if c["event_type"] == "rollback_check"]
    assert len(rollback_audit_entries) == 1
    assert rollback_audit_entries[0]["payload"]["skipped"] is True
    assert rollback_audit_entries[0]["payload"]["champion_history_id"] == 777


def test_run_tick_runs_real_rollback_check_when_nothing_promoted_this_tick():
    """Contrast case: when the gate rejects (or retrain wasn't triggered at
    all), check_rollback must still run normally - the skip only applies to
    the tick that actually promoted something.
    """
    batch_df = make_labeled_batch(n=100, seed=20)
    holdout_df = make_labeled_batch(n=100, seed=21)
    training_pool_df = make_labeled_batch(n=100, seed=22)
    conn = MagicMock()

    config = {
        "delayed_labels": {"delay_batches": 0},
        "gate": {
            **GATE_CONFIG,
            "retrain_drift_share_threshold": 0.3,
            "rollback_metric_drop_threshold": 0.03,
            "staleness_drift_share_delta_threshold": 0.3,
            "staleness_column_pvalue_delta_threshold": 0.3,
        },
    }

    with patch("src.orchestration.pipeline._production_cache") as mock_cache, \
         patch("src.orchestration.pipeline.check_retrain_trigger") as mock_trigger, \
         patch("src.orchestration.pipeline.release_due_labels"), \
         patch("src.orchestration.pipeline.score_model", return_value=np.full(100, 0.3)), \
         patch("src.orchestration.pipeline.check_rollback") as mock_rollback, \
         patch("src.orchestration.pipeline.write_audit_log"), \
         patch("src.orchestration.pipeline.inject_drift", side_effect=lambda df, idx, cfg: df), \
         patch("src.orchestration.pipeline.insert_predictions_bulk"):
        mock_cache.get.return_value = ("champion_model", "v1")
        mock_trigger.return_value = (False, {"drift_share": 0.0})  # no retrain this tick
        mock_rollback.return_value = {
            "rollback_triggered": False,
            "rollback_executed": False,
            "reference_stale": False,
        }

        result = run_tick(
            conn,
            current_batch=0,
            raw_batches=[batch_df],
            training_pool_df=training_pool_df,
            config=config,
            holdout_df=holdout_df,
        )

    assert "promoted_champion_history_id" not in result
    mock_rollback.assert_called_once()
    assert result["rollback"].get("skipped") is not True
