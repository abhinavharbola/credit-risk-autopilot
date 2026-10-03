from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import src.orchestration.pipeline as pipeline_mod
from src.model.features import ALL_COLUMNS, TARGET
from src.orchestration.pipeline import (
    build_expanded_training_pool,
    release_due_labels,
    retrain_and_gate,
    run_tick,
)
from tests.helpers import GATE_CONFIG, PIPELINE_CONFIG, make_labeled_batch


def labeled_rows(df):
    return [
        {"features": row.drop(TARGET).to_dict(), "true_label": int(row[TARGET])}
        for _, row in df.iterrows()
    ]


def test_retrain_and_gate_writes_audit_log_with_batch_on_rejection(monkeypatch):
    batch = make_labeled_batch()
    audit = []
    monkeypatch.setattr(pipeline_mod, "train_challenger", lambda df, run_name: ("r", "v2", "m"))
    monkeypatch.setattr(pipeline_mod, "score_model", lambda model, df: np.full(len(df), 0.5))
    monkeypatch.setattr(pipeline_mod, "write_audit_log", lambda conn, **kw: audit.append(kw))

    outcome = retrain_and_gate(
        MagicMock(), batch, "champ", batch, GATE_CONFIG, run_name="t", batch_index=7
    )

    assert outcome["gate_result"].promote is False
    assert len(audit) == 1
    assert audit[0]["event_type"] == "gate_evaluation"
    assert audit[0]["payload"]["batch"] == 7
    assert audit[0]["payload"]["promote"] is False


def test_retrain_and_gate_reports_promotion_when_challenger_clearly_wins(monkeypatch):
    batch = make_labeled_batch(n=400, seed=1, signal=0.5)
    audit = []
    monkeypatch.setattr(pipeline_mod, "train_challenger", lambda df, run_name: ("r", "v3", "chal"))

    def fake_score(model, df):
        y = df[TARGET].to_numpy()
        if model == "champ":
            return np.random.default_rng(0).uniform(0, 1, len(y))
        return np.where(y == 1, 0.9, 0.1) + np.random.default_rng(1).normal(0, 0.02, len(y))

    monkeypatch.setattr(pipeline_mod, "score_model", fake_score)
    monkeypatch.setattr(pipeline_mod, "write_audit_log", lambda conn, **kw: audit.append(kw))

    outcome = retrain_and_gate(MagicMock(), batch, "champ", batch, GATE_CONFIG, "t", 3)

    assert outcome["gate_result"].promote is True
    assert audit[0]["payload"]["promote"] is True


def test_expanded_pool_caps_base_sample_and_appends_labeled_rows(monkeypatch):
    base = make_labeled_batch(n=200, seed=0)
    extra = labeled_rows(make_labeled_batch(n=25, seed=99))
    monkeypatch.setattr(pipeline_mod, "get_labeled_predictions", lambda conn, **kw: extra)

    expanded = build_expanded_training_pool(MagicMock(), base, GATE_CONFIG)

    assert len(expanded) == GATE_CONFIG["retrain_base_pool_sample"] + len(extra)
    assert list(expanded.columns) == ALL_COLUMNS


def test_expanded_pool_does_not_cap_a_small_base_pool(monkeypatch):
    base = make_labeled_batch(n=30, seed=0)
    monkeypatch.setattr(pipeline_mod, "get_labeled_predictions", lambda conn, **kw: [])

    expanded = build_expanded_training_pool(MagicMock(), base, GATE_CONFIG)

    assert len(expanded) == 30


def test_expanded_pool_without_labels_is_still_capped_for_consistency(monkeypatch):
    base = make_labeled_batch(n=200, seed=0)
    monkeypatch.setattr(pipeline_mod, "get_labeled_predictions", lambda conn, **kw: [])

    expanded = build_expanded_training_pool(MagicMock(), base, GATE_CONFIG)

    assert len(expanded) == GATE_CONFIG["retrain_base_pool_sample"]


def test_expanded_pool_forwards_exclusion_and_limit_to_the_query(monkeypatch):
    seen = {}

    def fake_get(conn, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(pipeline_mod, "get_labeled_predictions", fake_get)

    build_expanded_training_pool(
        MagicMock(), make_labeled_batch(10), GATE_CONFIG, exclude_batch_ids=(7,)
    )

    assert seen["exclude_batch_ids"] == (7,)
    assert seen["limit"] == GATE_CONFIG["retrain_max_labeled_rows"]


def test_release_due_labels_rejects_row_count_mismatch(monkeypatch):
    monkeypatch.setattr(
        pipeline_mod, "get_predictions_for_batch", lambda conn, b, model_alias=None: [{"id": 1}]
    )
    with pytest.raises(ValueError):
        release_due_labels(MagicMock(), 0, make_labeled_batch(5))


def test_release_due_labels_maps_ids_to_labels_in_order(monkeypatch):
    batch = make_labeled_batch(4, seed=3)
    rows = [{"id": 10 + i} for i in range(4)]
    released = {}
    monkeypatch.setattr(
        pipeline_mod, "get_predictions_for_batch", lambda conn, b, model_alias=None: rows
    )
    monkeypatch.setattr(
        pipeline_mod, "release_labels_bulk", lambda conn, b, mapping: released.update(mapping)
    )

    release_due_labels(MagicMock(), 0, batch)

    assert released == {10 + i: int(v) for i, v in enumerate(batch[TARGET])}


class TickHarness:
    def __init__(self, monkeypatch, drift_share=0.5, last_gate_batch=None, promote=False):
        self.calls = []
        self.audit = []
        self.retrain_kwargs = None
        self.gate_promotes = promote
        monkeypatch.setattr(
            pipeline_mod, "reconcile_production_alias", lambda conn: self.calls.append("reconcile")
        )
        monkeypatch.setattr(
            pipeline_mod._production_cache, "get", lambda: (MagicMock(), "1")
        )
        monkeypatch.setattr(pipeline_mod, "score_batch_with_production", lambda *a, **k: None)
        monkeypatch.setattr(
            pipeline_mod,
            "check_retrain_trigger",
            lambda *a, **k: (
                drift_share >= GATE_CONFIG["retrain_drift_share_threshold"],
                {"drift_share": drift_share},
            ),
        )
        monkeypatch.setattr(pipeline_mod, "get_last_gate_batch", lambda conn: last_gate_batch)
        monkeypatch.setattr(
            pipeline_mod, "write_audit_log", lambda conn, event_type, payload: self.audit.append((event_type, payload))
        )
        monkeypatch.setattr(pipeline_mod, "release_due_labels", lambda *a, **k: None)
        monkeypatch.setattr(pipeline_mod, "_retrain_stage", self.fake_retrain)
        monkeypatch.setattr(
            pipeline_mod,
            "_rollback_stage",
            lambda *a, **k: {"rollback_triggered": False, "rollback_executed": False},
        )

    def fake_retrain(self, conn, current_batch, due_batch_index, *a, **k):
        self.retrain_kwargs = {"current_batch": current_batch, "due_batch_index": due_batch_index}
        return {"gate": {"promote": self.gate_promotes}, "promoted_champion_history_id": 5 if self.gate_promotes else None}


def tick(current_batch):
    batches = [make_labeled_batch(20, seed=i) for i in range(current_batch + 1)]
    return run_tick(
        MagicMock(), current_batch, batches, make_labeled_batch(20), PIPELINE_CONFIG, make_labeled_batch(20)
    )


def test_tick_reconciles_alias_before_anything_else(monkeypatch):
    harness = TickHarness(monkeypatch, drift_share=0.0)
    tick(4)
    assert harness.calls == ["reconcile"]


def test_no_retrain_before_the_first_labels_are_due(monkeypatch):
    harness = TickHarness(monkeypatch, drift_share=0.9)
    result = tick(1)

    assert result["drift_detected"] is True
    assert result["retrain_triggered"] is False
    assert harness.retrain_kwargs is None
    drift_payload = [p for t, p in harness.audit if t == "drift_check"][0]
    assert drift_payload["retrain_triggered"] is False
    assert drift_payload["drift_detected"] is True


def test_retrain_runs_on_drift_once_labels_are_due(monkeypatch):
    harness = TickHarness(monkeypatch, drift_share=0.9)
    result = tick(4)

    assert result["retrain_triggered"] is True
    assert harness.retrain_kwargs == {"current_batch": 4, "due_batch_index": 1}


def test_cooldown_suppresses_back_to_back_retrains(monkeypatch):
    harness = TickHarness(monkeypatch, drift_share=0.9, last_gate_batch=3)
    result = tick(4)

    assert result["drift_detected"] is True
    assert result["retrain_triggered"] is False
    assert harness.retrain_kwargs is None


def test_retrain_resumes_after_the_cooldown_elapses(monkeypatch):
    harness = TickHarness(monkeypatch, drift_share=0.9, last_gate_batch=3)
    result = tick(6)

    assert result["retrain_triggered"] is True
    assert harness.retrain_kwargs["current_batch"] == 6


def test_no_retrain_without_drift(monkeypatch):
    harness = TickHarness(monkeypatch, drift_share=0.0)
    result = tick(5)

    assert result["retrain_triggered"] is False
    assert harness.retrain_kwargs is None


def test_tick_skips_rollback_check_on_the_tick_it_promotes(monkeypatch):
    harness = TickHarness(monkeypatch, drift_share=0.9, promote=True)
    result = tick(4)

    assert result["promoted_champion_history_id"] == 5
    assert result["rollback"]["skipped"] is True
    skipped = [p for t, p in harness.audit if t == "rollback_check"][0]
    assert skipped["skipped"] is True


def test_tick_runs_real_rollback_check_when_nothing_promoted(monkeypatch):
    TickHarness(monkeypatch, drift_share=0.0)
    result = tick(5)

    assert "skipped" not in result["rollback"]
    assert "promoted_champion_history_id" not in result


def test_pool_frame_is_returned_unmodified_type(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "get_labeled_predictions", lambda conn, **kw: [])
    out = build_expanded_training_pool(MagicMock(), make_labeled_batch(10), GATE_CONFIG)
    assert isinstance(out, pd.DataFrame)
