from unittest.mock import MagicMock

import src.orchestration.clock as clock_mod
from src.orchestration.clock import claim_and_run_tick


def test_losing_racer_never_calls_run_tick_or_writes_audit(monkeypatch):
    claims = {"n": 0}
    run_tick_calls = []
    audit = []

    monkeypatch.setattr(clock_mod, "get_pipeline_state", lambda conn: {"version": 1, "current_batch": 3})

    def fake_advance(conn, expected_version):
        claims["n"] += 1
        return claims["n"] == 1

    monkeypatch.setattr(clock_mod, "advance_pipeline_state", fake_advance)
    monkeypatch.setattr(
        clock_mod, "run_tick", lambda conn, batch, *a: run_tick_calls.append(batch) or {"batch": batch}
    )
    monkeypatch.setattr(clock_mod, "write_audit_log", lambda conn, t, p: audit.append((t, p)))

    batches = [None] * 10
    winner = claim_and_run_tick(MagicMock(), batches, None, {}, None)
    loser = claim_and_run_tick(MagicMock(), batches, None, {}, None)

    assert winner == {"batch": 3}
    assert loser is None
    assert run_tick_calls == [3]
    assert audit == [("clock_advance", {"batch": 3, "version": 2})]


def test_claim_returns_status_when_past_end_of_dataset(monkeypatch):
    monkeypatch.setattr(clock_mod, "get_pipeline_state", lambda conn: {"version": 1, "current_batch": 10})
    monkeypatch.setattr(
        clock_mod, "advance_pipeline_state", lambda *a: (_ for _ in ()).throw(AssertionError("no claim"))
    )

    result = claim_and_run_tick(MagicMock(), [None] * 10, None, {}, None)

    assert result["status"] == "past_end_of_dataset"
