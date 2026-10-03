from unittest.mock import MagicMock

import numpy as np
import pytest

import src.orchestration.promote as promote_mod
from src.model.features import TARGET
from src.orchestration.promote import check_rollback, find_previous_champion
from tests.helpers import GATE_CONFIG, make_labeled_batch


def test_find_previous_champion_skips_rolled_back_and_future_entries():
    history = [
        {"id": 1, "rolled_back_at": None},
        {"id": 2, "rolled_back_at": "2026-01-01"},
        {"id": 3, "rolled_back_at": None},
        {"id": 4, "rolled_back_at": None},
    ]
    assert find_previous_champion(history, current_champion_history_id=4)["id"] == 3


def test_find_previous_champion_skips_to_older_hop_if_immediate_predecessor_was_rolled_back():
    history = [
        {"id": 1, "rolled_back_at": None},
        {"id": 2, "rolled_back_at": "2026-01-01"},
        {"id": 3, "rolled_back_at": "2026-01-02"},
    ]
    assert find_previous_champion(history, current_champion_history_id=3)["id"] == 1


def test_find_previous_champion_returns_none_when_no_valid_candidate():
    assert find_previous_champion([{"id": 1, "rolled_back_at": None}], 1) is None


class RollbackHarness:
    def __init__(self, monkeypatch, current, history, previous_prob, stale=False):
        self.audit = []
        self.rolled_back = []
        self.alias_calls = []
        self.stale_marks = []
        monkeypatch.setattr(promote_mod, "get_latest_champion", lambda conn: current)
        monkeypatch.setattr(promote_mod, "get_champion_history", lambda conn: history)
        monkeypatch.setattr(promote_mod, "compute_fingerprint", lambda *a, **k: {"drift_share": 0.1})
        monkeypatch.setattr(promote_mod, "check_fingerprint_staleness", lambda *a, **k: stale)
        monkeypatch.setattr(
            promote_mod, "mark_reference_stale", lambda conn, i, s: self.stale_marks.append((i, s))
        )
        monkeypatch.setattr(
            promote_mod, "write_audit_log", lambda conn, t, p: self.audit.append((t, p))
        )
        monkeypatch.setattr(
            promote_mod, "record_rollback", lambda conn, i, v: self.rolled_back.append((i, v))
        )
        monkeypatch.setattr(promote_mod, "set_production_alias", self.alias_calls.append)
        monkeypatch.setattr(promote_mod, "load_model_version", lambda v: f"model-{v}")
        monkeypatch.setattr(promote_mod, "score", lambda model, df: previous_prob)


def make_history():
    return [
        {"id": 1, "model_version": "1", "rolled_back_at": None},
        {
            "id": 2,
            "model_version": "2",
            "rolled_back_at": None,
            "window_metrics": {"auc_pr": 0.5},
            "drift_fingerprint": {"drift_share": 0.1},
        },
    ]


def run_check(harness_current, live_prob, live_metric, batch):
    return check_rollback(
        MagicMock(),
        batch,
        live_prob,
        {"auc_pr": live_metric},
        batch,
        {**GATE_CONFIG, "bootstrap_resamples": 300},
    )


def test_rollback_skipped_and_marked_stale_when_reference_is_stale(monkeypatch):
    history = make_history()
    batch = make_labeled_batch(200, seed=1)
    harness = RollbackHarness(monkeypatch, history[1], history, np.zeros(200), stale=True)

    result = run_check(history[1], np.zeros(200), 0.0, batch)

    assert result["reference_stale"] is True
    assert result["rollback_triggered"] is False
    assert harness.stale_marks == [(2, True)]
    assert harness.alias_calls == []
    assert harness.audit[0][0] == "rollback_check"


def test_no_degradation_means_no_rollback_and_no_candidate_scoring(monkeypatch):
    history = make_history()
    batch = make_labeled_batch(200, seed=2)
    harness = RollbackHarness(monkeypatch, history[1], history, np.zeros(200))

    result = run_check(history[1], np.zeros(200), 0.49, batch)

    assert result["degradation_flagged"] is False
    assert result["rollback_triggered"] is False
    assert harness.rolled_back == []


def test_degradation_without_earlier_champion_flags_but_does_not_roll_back(monkeypatch):
    history = [make_history()[1]]
    batch = make_labeled_batch(200, seed=3)
    harness = RollbackHarness(monkeypatch, history[0], history, np.zeros(200))

    result = run_check(history[0], np.zeros(200), 0.1, batch)

    assert result["degradation_flagged"] is True
    assert result["rollback_triggered"] is False
    assert "no earlier champion" in result["reason"]
    assert harness.alias_calls == []


def test_degradation_with_significantly_better_earlier_champion_rolls_back(monkeypatch):
    history = make_history()
    batch = make_labeled_batch(600, seed=4, signal=0.5)
    y = batch[TARGET].to_numpy()
    good = np.where(y == 1, 0.9, 0.1) + np.random.default_rng(0).normal(0, 0.02, len(y))
    bad = np.random.default_rng(1).uniform(0, 1, len(y))
    harness = RollbackHarness(monkeypatch, history[1], history, good)

    result = run_check(history[1], bad, 0.1, batch)

    assert result["rollback_triggered"] is True
    assert result["rollback_executed"] is True
    assert harness.rolled_back == [(2, "1")]
    assert harness.alias_calls == ["1"]
    assert [t for t, _ in harness.audit] == ["rollback_check", "rollback"]
    assert harness.audit[1][1]["rolled_back_from"] == "2"
    assert harness.audit[1][1]["rolled_back_to"] == "1"


def test_degradation_with_equally_good_earlier_champion_does_not_roll_back(monkeypatch):
    history = make_history()
    batch = make_labeled_batch(300, seed=5, signal=0.3)
    same = np.random.default_rng(2).uniform(0, 1, len(batch))
    harness = RollbackHarness(monkeypatch, history[1], history, same)

    result = run_check(history[1], same, 0.1, batch)

    assert result["degradation_flagged"] is True
    assert result["rollback_triggered"] is False
    assert harness.rolled_back == []
    assert harness.alias_calls == []


def test_alias_is_moved_last_on_promotion(monkeypatch):
    order = []
    monkeypatch.setattr(promote_mod, "compute_fingerprint", lambda *a, **k: order.append("fp") or {})
    monkeypatch.setattr(
        promote_mod, "insert_champion_history", lambda conn, row: order.append("insert") or 9
    )
    monkeypatch.setattr(promote_mod, "write_audit_log", lambda *a, **k: order.append("audit"))
    monkeypatch.setattr(promote_mod, "set_production_alias", lambda v: order.append("alias"))

    batch = make_labeled_batch(20)
    history_id = promote_mod.promote_challenger(
        MagicMock(), "5", {"auc_pr": 0.4}, {"auc_pr": 0.5}, batch, batch
    )

    assert history_id == 9
    assert order == ["fp", "insert", "audit", "alias"]


def test_reconcile_resets_alias_to_the_database_champion(monkeypatch):
    calls = []
    audit = []
    monkeypatch.setattr(promote_mod, "get_latest_champion", lambda conn: {"model_version": "3"})
    monkeypatch.setattr(promote_mod, "current_production_version", lambda: "4")
    monkeypatch.setattr(promote_mod, "set_production_alias", calls.append)
    monkeypatch.setattr(promote_mod, "write_audit_log", lambda conn, t, p: audit.append((t, p)))

    assert promote_mod.reconcile_production_alias(MagicMock()) is True
    assert calls == ["3"]
    assert audit == [("alias_reconciled", {"expected_version": "3", "found_version": "4"})]


def test_reconcile_is_a_noop_when_alias_already_matches(monkeypatch):
    monkeypatch.setattr(promote_mod, "get_latest_champion", lambda conn: {"model_version": "3"})
    monkeypatch.setattr(promote_mod, "current_production_version", lambda: "3")
    monkeypatch.setattr(
        promote_mod, "set_production_alias", lambda v: pytest.fail("alias should not move")
    )
    assert promote_mod.reconcile_production_alias(MagicMock()) is False


def test_reconcile_is_a_noop_without_any_champion(monkeypatch):
    monkeypatch.setattr(promote_mod, "get_latest_champion", lambda conn: None)
    assert promote_mod.reconcile_production_alias(MagicMock()) is False
