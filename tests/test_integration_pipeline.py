import importlib.util
import os
import sys

import mlflow
import pytest
from sqlalchemy import create_engine, text

import src.orchestration.pipeline as pipeline_mod
from src.data.split import build_pretrain_batches
from src.db import repository as repo
from src.db.connection import get_connection, normalize_database_url, run_migrations
from src.orchestration.clock import claim_and_run_tick
from src.orchestration.promote import current_production_version, set_production_alias
from src.utils.config import MODEL_NAME, REPO_ROOT
from tests.helpers import GATE_CONFIG, make_labeled_batch

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not DATABASE_URL, reason="TEST_DATABASE_URL is not set, skipping end-to-end pipeline test"
)

TABLES = ("audit_log", "champion_history", "predictions", "pipeline_state")

CONFIG = {
    "persistent_drift": {
        "start_batch": 2,
        "columns": {
            "RevolvingUtilizationOfUnsecuredLines": {"shift": 0.6, "scale": 1.0},
            "DebtRatio": {"shift": 0.6, "scale": 1.0},
            "MonthlyIncome": {"shift": -0.5, "scale": 1.0},
        },
    },
    "temporary_concept_drift": {
        "start_batch": 100,
        "end_batch": 101,
        "blend_ratio": 0.5,
        "columns": ["DebtRatio"],
    },
    "delayed_labels": {"delay_batches": 2},
    "gate": {**GATE_CONFIG, "retrain_base_pool_sample": 300, "retrain_max_labeled_rows": 800},
}


def load_demo_module():
    spec = importlib.util.spec_from_file_location(
        "run_demo_loop_under_test", REPO_ROOT / "scripts" / "run_demo_loop.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def world(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.chdir(tmp_path)
    import src.db.connection as connection

    monkeypatch.setattr(connection, "_engine", None)
    engine = create_engine(normalize_database_url(DATABASE_URL))
    with engine.begin() as conn:
        for table in TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
    run_migrations()

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setattr(
        pipeline_mod, "_production_cache", pipeline_mod.AliasedModelCache(MODEL_NAME, "production")
    )

    base_pool = make_labeled_batch(2000, seed=1, signal=0.5)
    holdout = make_labeled_batch(600, seed=2, signal=0.5)
    stream = make_labeled_batch(200 * 12, seed=3, signal=0.5)
    batches = build_pretrain_batches(stream, batch_size=200, seed=0)

    demo = load_demo_module()
    demo.bootstrap_champion(base_pool, holdout, CONFIG["gate"])
    yield {"engine": engine, "base_pool": base_pool, "holdout": holdout, "batches": batches}
    engine.dispose()


def run_ticks(world, n):
    results = []
    for _ in range(n):
        with get_connection() as conn:
            results.append(
                claim_and_run_tick(
                    conn, world["batches"], world["base_pool"], CONFIG, world["holdout"]
                )
            )
    return results


def audit_rows(world, event_type):
    with world["engine"].begin() as conn:
        return list(reversed(repo.get_audit_log(conn, event_type=event_type, limit=1000)))


def test_gate_window_never_leaks_into_challenger_training_and_cooldown_holds(world, monkeypatch):
    captured = []
    original_train = pipeline_mod.train_challenger

    def spy(train_df, run_name):
        captured.append((run_name, train_df.copy()))
        return original_train(train_df, run_name=run_name)

    monkeypatch.setattr(pipeline_mod, "train_challenger", spy)

    results = run_ticks(world, 10)

    assert all(r is not None and "status" not in r for r in results)
    assert captured, "drift should have triggered at least one retrain"

    for run_name, train_df in captured:
        current_batch = int(run_name.rsplit("-", 1)[1])
        due = current_batch - CONFIG["delayed_labels"]["delay_batches"]
        due_df = pipeline_mod.inject_drift(world["batches"][due], due, CONFIG)
        overlap = set(train_df["RevolvingUtilizationOfUnsecuredLines"]) & set(
            due_df["RevolvingUtilizationOfUnsecuredLines"]
        )
        assert not overlap, f"gate batch {due} leaked into training at tick {current_batch}"

    gate_batches = [r["event_payload"]["batch"] for r in audit_rows(world, "gate_evaluation")]
    assert gate_batches == sorted(gate_batches)
    assert all(b - a >= CONFIG["gate"]["retrain_cooldown_batches"] for a, b in zip(gate_batches, gate_batches[1:]))


def test_disjoint_stream_and_labeled_rows_cover_only_earlier_batches(world):
    run_ticks(world, 6)

    with world["engine"].begin() as conn:
        labeled_batches = {
            r[0]
            for r in conn.execute(
                text("SELECT DISTINCT batch_id FROM predictions WHERE true_label IS NOT NULL")
            )
        }
    assert labeled_batches == {0, 1, 2, 3}


def test_every_tick_records_a_clock_advance_and_a_drift_check(world):
    run_ticks(world, 5)

    assert [r["event_payload"]["batch"] for r in audit_rows(world, "clock_advance")] == [0, 1, 2, 3, 4]
    drift_checks = audit_rows(world, "drift_check")
    assert len(drift_checks) == 5
    assert all("drift_detected" in r["event_payload"] for r in drift_checks)
    assert all(
        not r["event_payload"]["retrain_triggered"]
        for r in drift_checks
        if r["event_payload"]["batch"] < CONFIG["delayed_labels"]["delay_batches"]
    )


def test_a_stale_alias_is_reconciled_to_the_database_champion_on_the_next_tick(world):
    run_ticks(world, 1)
    with world["engine"].begin() as conn:
        db_version = str(repo.get_latest_champion(conn)["model_version"])

    client = mlflow.MlflowClient()
    original_champion_version = db_version
    client.create_model_version(MODEL_NAME, source=client.get_model_version(MODEL_NAME, db_version).source)
    latest = max(int(mv.version) for mv in client.search_model_versions(f"name='{MODEL_NAME}'"))
    set_production_alias(str(latest))
    assert current_production_version() != original_champion_version

    run_ticks(world, 1)

    assert current_production_version() == original_champion_version
    reconciled = audit_rows(world, "alias_reconciled")
    assert len(reconciled) == 1
    assert reconciled[0]["event_payload"]["expected_version"] == original_champion_version


def test_a_failed_tick_rolls_back_the_claim_and_leaves_no_partial_state(world, monkeypatch):
    original_score = pipeline_mod.score_batch_with_production

    def boom(*args, **kwargs):
        raise RuntimeError("scoring failed mid-tick")

    monkeypatch.setattr(pipeline_mod, "score_batch_with_production", boom)

    with pytest.raises(RuntimeError):
        run_ticks(world, 1)

    with world["engine"].begin() as conn:
        state = repo.get_pipeline_state(conn)
    assert state["current_batch"] == 0
    assert state["version"] == 0
    assert audit_rows(world, "clock_advance") == []

    monkeypatch.setattr(pipeline_mod, "score_batch_with_production", original_score)
    retry = run_ticks(world, 1)
    assert retry[0]["batch"] == 0


def test_promotions_and_rollbacks_keep_alias_and_database_in_agreement(world):
    run_ticks(world, 12)

    with world["engine"].begin() as conn:
        latest = repo.get_latest_champion(conn)
    assert current_production_version() == str(latest["model_version"])

    gate_rows = audit_rows(world, "gate_evaluation")
    promotions = audit_rows(world, "promotion")
    promoted_gates = [r for r in gate_rows if r["event_payload"]["promote"]]
    assert len(promotions) == 1 + len(promoted_gates)
