"""Tests for the single-writer, optimistic-concurrency pipeline_state advance
(4.6a), and for claim_and_run_tick's claim-before-work ordering: a losing
racer must produce zero side effects, not just a no-op clock advance.
"""

import tests._stubs  # noqa: F401  (must run before src imports below)

from unittest.mock import MagicMock

from src.db.repository import advance_pipeline_state
from src.orchestration.clock import claim_and_run_tick


def make_fake_conn(current_version: int, current_batch: int = 0):
    """Fake Connection whose execute() mimics the UPDATE ... WHERE version =
    :expected_version RETURNING ... behavior used by advance_pipeline_state.
    """
    state = {"version": current_version, "current_batch": current_batch}

    def fake_execute(stmt, params=None):
        result = MagicMock()
        if params and "expected_version" in params:
            if params["expected_version"] == state["version"]:
                state["version"] += 1
                state["current_batch"] += 1
                result.first.return_value = (state["current_batch"], state["version"])
            else:
                result.first.return_value = None
        return result

    conn = MagicMock()
    conn.execute.side_effect = fake_execute
    return conn, state


def test_advance_succeeds_when_version_matches():
    conn, state = make_fake_conn(current_version=5)
    advanced = advance_pipeline_state(conn, expected_version=5)
    assert advanced is True
    assert state["version"] == 6


def test_advance_is_noop_when_version_stale():
    conn, state = make_fake_conn(current_version=5)
    first_call = advance_pipeline_state(conn, expected_version=5)
    second_call = advance_pipeline_state(conn, expected_version=5)
    assert first_call is True
    assert second_call is False
    assert state["version"] == 6


def test_concurrent_invocation_never_double_advances():
    conn, state = make_fake_conn(current_version=0)
    results = [advance_pipeline_state(conn, expected_version=0) for _ in range(5)]
    assert results.count(True) == 1
    assert results.count(False) == 4
    assert state["version"] == 1


def test_losing_racer_never_calls_run_tick():
    """The core 4.6a guarantee: claim happens before any tick work, so a
    losing racer produces zero side effects (predictions, audit_log writes),
    not just a no-op clock advance.
    """
    import src.orchestration.clock as clock_mod

    conn = MagicMock()

    call_count = {"n": 0}

    def fake_get_pipeline_state(conn):
        return {"version": 1, "current_batch": 3}

    def fake_advance_pipeline_state(conn, expected_version):
        call_count["n"] += 1
        # first caller wins, every subsequent caller (racing on the same
        # expected_version) loses
        return call_count["n"] == 1

    run_tick_calls = []

    def fake_run_tick(conn, current_batch, raw_batches, training_pool_df, config):
        run_tick_calls.append(current_batch)
        return {"batch": current_batch}

    orig_get_state = clock_mod.get_pipeline_state
    orig_advance = clock_mod.advance_pipeline_state
    orig_run_tick = clock_mod.run_tick
    clock_mod.get_pipeline_state = fake_get_pipeline_state
    clock_mod.advance_pipeline_state = fake_advance_pipeline_state
    clock_mod.run_tick = fake_run_tick

    try:
        raw_batches = [None] * 10
        result_a = claim_and_run_tick(conn, raw_batches, training_pool_df=None, config={})
        result_b = claim_and_run_tick(conn, raw_batches, training_pool_df=None, config={})
    finally:
        clock_mod.get_pipeline_state = orig_get_state
        clock_mod.advance_pipeline_state = orig_advance
        clock_mod.run_tick = orig_run_tick

    assert result_a == {"batch": 3}
    assert result_b is None  # losing racer: no tick result at all
    assert run_tick_calls == [3]  # run_tick was called exactly once, not twice


def test_claim_returns_status_when_past_end_of_dataset():
    import src.orchestration.clock as clock_mod

    conn = MagicMock()
    orig_get_state = clock_mod.get_pipeline_state
    clock_mod.get_pipeline_state = lambda conn: {"version": 1, "current_batch": 10}

    try:
        result = claim_and_run_tick(conn, raw_batches=[None] * 10, training_pool_df=None, config={})
    finally:
        clock_mod.get_pipeline_state = orig_get_state

    assert result["status"] == "past_end_of_dataset"
