import os
import threading

import pytest
from sqlalchemy import create_engine, text

from src.db import repository as repo
from src.db.connection import normalize_database_url, run_migrations

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not DATABASE_URL, reason="TEST_DATABASE_URL is not set, skipping Postgres integration tests"
)

TABLES = ("audit_log", "champion_history", "predictions", "pipeline_state")


@pytest.fixture
def engine(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    import src.db.connection as connection

    monkeypatch.setattr(connection, "_engine", None)
    eng = create_engine(normalize_database_url(DATABASE_URL))
    with eng.begin() as conn:
        for table in TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
    run_migrations()
    yield eng
    eng.dispose()


def prediction_row(batch_id, i, alias="production"):
    return {
        "batch_id": batch_id,
        "model_alias": alias,
        "model_version": "1",
        "features": {"age": 40 + i, "DebtRatio": 0.5},
        "predicted_prob": 0.3,
        "predicted_label": 0,
    }


def test_migrations_are_idempotent_and_seed_a_single_state_row(engine):
    run_migrations()
    run_migrations()
    with engine.begin() as conn:
        state = repo.get_pipeline_state(conn)
    assert state["current_batch"] == 0
    assert state["version"] == 0


def test_advance_claims_exactly_once_for_the_same_expected_version(engine):
    with engine.begin() as conn:
        assert repo.advance_pipeline_state(conn, 0) is True
    with engine.begin() as conn:
        assert repo.advance_pipeline_state(conn, 0) is False
        assert repo.get_pipeline_state(conn)["current_batch"] == 1


def test_racer_does_not_block_while_the_winner_holds_the_row_lock(engine):
    winner = engine.connect()
    winner_tx = winner.begin()
    assert repo.advance_pipeline_state(winner, 0) is True

    outcome = {}

    def racer():
        with engine.begin() as conn:
            conn.execute(text("SET LOCAL lock_timeout = '3s'"))
            outcome["claimed"] = repo.advance_pipeline_state(conn, 0)

    thread = threading.Thread(target=racer)
    thread.start()
    thread.join(timeout=8)

    assert not thread.is_alive()
    assert outcome["claimed"] is False
    winner_tx.commit()
    winner.close()
    with engine.begin() as conn:
        assert repo.get_pipeline_state(conn)["current_batch"] == 1


def test_many_concurrent_claims_produce_a_single_winner(engine):
    results = []
    barrier = threading.Barrier(6)

    def claim():
        barrier.wait()
        with engine.begin() as conn:
            results.append(repo.advance_pipeline_state(conn, 0))

    threads = [threading.Thread(target=claim) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 1
    with engine.begin() as conn:
        assert repo.get_pipeline_state(conn)["version"] == 1


def test_release_labels_and_labeled_query_filters_ordering_and_limit(engine):
    with engine.begin() as conn:
        for batch_id in range(3):
            repo.insert_predictions_bulk(conn, [prediction_row(batch_id, i) for i in range(4)])
        repo.insert_predictions_bulk(conn, [prediction_row(0, 99, alias="challenger")])
        for batch_id in range(3):
            rows = repo.get_predictions_for_batch(conn, batch_id, model_alias="production")
            repo.release_labels_bulk(
                conn, batch_id, {r["id"]: (batch_id + r["id"]) % 2 for r in rows}
            )

    with engine.begin() as conn:
        everything = repo.get_labeled_predictions(conn)
        without_two = repo.get_labeled_predictions(conn, exclude_batch_ids=(2,))
        limited = repo.get_labeled_predictions(conn, limit=5)

    assert len(everything) == 12
    assert len(without_two) == 8
    assert all(isinstance(r["features"], dict) for r in everything)
    assert len(limited) == 5
    assert limited[0]["features"]["age"] == 43
    assert all(r["features"]["age"] != 99 for r in everything)


def test_unlabeled_rows_are_never_returned(engine):
    with engine.begin() as conn:
        repo.insert_predictions_bulk(conn, [prediction_row(0, i) for i in range(3)])
        assert repo.get_labeled_predictions(conn) == []


def test_champion_history_latest_and_rollback_ordering(engine):
    with engine.begin() as conn:
        first = repo.insert_champion_history(
            conn,
            {"model_version": "1", "holdout_metrics": {"m": 1}, "window_metrics": {"m": 1}, "drift_fingerprint": {}},
        )
        second = repo.insert_champion_history(
            conn,
            {"model_version": "2", "holdout_metrics": {"m": 2}, "window_metrics": {"m": 2}, "drift_fingerprint": {"drift_share": 0.2}},
        )
        assert repo.get_latest_champion(conn)["id"] == second
        repo.record_rollback(conn, second, "1")
        latest = repo.get_latest_champion(conn)
        history = repo.get_champion_history(conn)

    assert latest["id"] == first
    assert [h["id"] for h in history] == [first, second]
    assert history[1]["rolled_back_to_version"] == "1"
    assert history[1]["drift_fingerprint"] == {"drift_share": 0.2}


def test_audit_payload_filter_is_applied_before_the_limit(engine):
    with engine.begin() as conn:
        for i in range(12):
            repo.write_audit_log(conn, "gate_evaluation", {"promote": i % 6 == 0, "batch": i})
        rejected = repo.get_audit_log(
            conn, event_type="gate_evaluation", limit=10, payload_equals={"promote": "false"}
        )
        promoted = repo.get_audit_log(
            conn, event_type="gate_evaluation", limit=10, payload_equals={"promote": "true"}
        )

    assert len(rejected) == 10
    assert all(r["event_payload"]["promote"] is False for r in rejected)
    assert len(promoted) == 2


def test_last_gate_batch_ignores_rows_without_a_batch_and_takes_the_latest(engine):
    with engine.begin() as conn:
        assert repo.get_last_gate_batch(conn) is None
        repo.write_audit_log(conn, "gate_evaluation", {"promote": False})
        assert repo.get_last_gate_batch(conn) is None
        repo.write_audit_log(conn, "gate_evaluation", {"promote": False, "batch": 4})
        repo.write_audit_log(conn, "gate_evaluation", {"promote": False, "batch": 7})
        repo.write_audit_log(conn, "drift_check", {"batch": 99})
        assert repo.get_last_gate_batch(conn) == 7


def test_audit_payload_accepts_numpy_scalars(engine):
    import numpy as np

    with engine.begin() as conn:
        repo.write_audit_log(
            conn, "gate_evaluation", {"x": np.float64(0.5), "flag": np.bool_(True), "n": np.int64(3)}
        )
        row = repo.get_audit_log(conn, event_type="gate_evaluation")[0]

    assert row["event_payload"] == {"x": 0.5, "flag": True, "n": 3}
