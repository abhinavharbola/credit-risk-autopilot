import numpy as np
import pandas as pd
import pytest

import src.drift.detect as detect_mod
from src.drift.detect import (
    DriftFingerprintError,
    _reduce_to_fingerprint,
    check_fingerprint_staleness,
    check_retrain_trigger,
    compute_fingerprint,
)


def ks_output(share, pvalues, method="ks"):
    metrics = [
        {
            "metric_name": "DriftedColumnsCount",
            "config": {"type": "evidently:metric_v2:DriftedColumnsCount", "drift_share": 0.5},
            "value": {"count": share * len(pvalues), "share": share},
        }
    ]
    for column, value in pvalues.items():
        metrics.append(
            {
                "metric_name": f"ValueDrift(column={column})",
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": column,
                    "method": method,
                    "threshold": 0.05,
                },
                "value": value,
            }
        )
    return {"metrics": metrics, "tests": []}


def test_reduce_extracts_drift_share_scores_and_drifted_columns():
    raw = ks_output(1 / 3, {"DebtRatio": 4.5e-66, "MonthlyIncome": 0.18, "age": 0.79})
    fingerprint = _reduce_to_fingerprint(raw)
    assert fingerprint["drift_share"] == pytest.approx(1 / 3)
    assert fingerprint["column_drift_scores"]["MonthlyIncome"] == 0.18
    assert fingerprint["drifted_columns"] == ["DebtRatio"]


def test_reduce_treats_distance_methods_as_drifted_above_threshold():
    raw = ks_output(0.5, {"DebtRatio": 0.4, "age": 0.02}, method="Wasserstein distance (normed)")
    fingerprint = _reduce_to_fingerprint(raw)
    assert fingerprint["drifted_columns"] == ["DebtRatio"]


def test_reduce_handles_missing_metrics_key_gracefully():
    assert _reduce_to_fingerprint({}) == {
        "drift_share": None,
        "column_drift_scores": {},
        "drifted_columns": [],
    }


def test_check_retrain_trigger_raises_instead_of_silently_not_triggering(monkeypatch):
    frame = pd.DataFrame({"DebtRatio": [0.1, 0.2]})
    monkeypatch.setattr(
        detect_mod,
        "compute_fingerprint",
        lambda *a, **k: {"drift_share": None, "column_drift_scores": {}, "drifted_columns": []},
    )
    with pytest.raises(DriftFingerprintError):
        check_retrain_trigger(frame, frame, drift_share_threshold=0.3)


@pytest.mark.parametrize("share,expected", [(0.3, True), (0.29, False), (0.9, True), (0.0, False)])
def test_check_retrain_trigger_compares_share_to_threshold(monkeypatch, share, expected):
    frame = pd.DataFrame({"DebtRatio": [0.1, 0.2]})
    monkeypatch.setattr(
        detect_mod,
        "compute_fingerprint",
        lambda *a, **k: {"drift_share": share, "column_drift_scores": {}, "drifted_columns": []},
    )
    triggered, fingerprint = check_retrain_trigger(frame, frame, drift_share_threshold=0.3)
    assert triggered is expected
    assert fingerprint["drift_share"] == share


def test_real_evidently_uses_ks_pvalues_even_for_a_large_reference():
    rng = np.random.default_rng(0)
    columns = ["DebtRatio", "MonthlyIncome", "age"]

    def frame(n, shift=0.0):
        return pd.DataFrame(
            {
                "DebtRatio": rng.pareto(1.2, n) * 100 + shift * 100,
                "MonthlyIncome": rng.lognormal(8.5, 0.6, n),
                "age": rng.integers(21, 90, n).astype(float),
            }
        )

    reference = frame(6000)
    stable = compute_fingerprint(frame(200), reference, columns=columns)
    drifted = compute_fingerprint(frame(200, shift=1.0), reference, columns=columns)

    assert all(0.0 <= v <= 1.0 for v in stable["column_drift_scores"].values())
    assert "DebtRatio" not in stable["drifted_columns"] or stable["drift_share"] <= 1 / 3
    assert "DebtRatio" in drifted["drifted_columns"]
    assert drifted["drift_share"] >= 1 / 3


def test_staleness_flags_large_drift_share_delta():
    then = {"drift_share": 0.1, "column_drift_scores": {}, "drifted_columns": []}
    now = {"drift_share": 0.5, "column_drift_scores": {}, "drifted_columns": []}
    assert check_fingerprint_staleness(then, now, 0.3, 0.3) is True


def test_staleness_false_when_regime_is_unchanged():
    columns = {c: 0.5 for c in "abcdefghij"}
    then = {"drift_share": 0.2, "column_drift_scores": columns, "drifted_columns": ["a", "b"]}
    now = {"drift_share": 0.2, "column_drift_scores": columns, "drifted_columns": ["a", "b"]}
    assert check_fingerprint_staleness(then, now, 0.3, 0.3) is False


def test_staleness_flags_different_drifted_columns_even_if_share_is_equal():
    columns = {c: 0.5 for c in "abcdefghij"}
    then = {"drift_share": 0.3, "column_drift_scores": columns, "drifted_columns": ["a", "b", "c"]}
    now = {"drift_share": 0.3, "column_drift_scores": columns, "drifted_columns": ["d", "e", "f"]}
    assert check_fingerprint_staleness(then, now, 0.3, 0.3) is True


def test_staleness_is_not_triggered_by_pvalue_noise_alone():
    rng = np.random.default_rng(0)
    columns = list("abcdefghij")
    stale_count = 0
    for _ in range(200):
        then_p = {c: float(rng.uniform()) for c in columns}
        now_p = {c: float(rng.uniform()) for c in columns}
        then = {
            "drift_share": sum(p < 0.05 for p in then_p.values()) / 10,
            "column_drift_scores": then_p,
            "drifted_columns": [c for c, p in then_p.items() if p < 0.05],
        }
        now = {
            "drift_share": sum(p < 0.05 for p in now_p.values()) / 10,
            "column_drift_scores": now_p,
            "drifted_columns": [c for c, p in now_p.items() if p < 0.05],
        }
        stale_count += check_fingerprint_staleness(then, now, 0.3, 0.3)
    assert stale_count / 200 < 0.05


def test_staleness_ignores_column_check_for_legacy_fingerprints():
    then = {"drift_share": 0.2, "column_drift_scores": {"a": 0.1}}
    now = {"drift_share": 0.21, "column_drift_scores": {"a": 0.9}}
    assert check_fingerprint_staleness(then, now, 0.3, 0.3) is False


def test_real_evidently_handles_low_cardinality_and_constant_columns():
    rng = np.random.default_rng(1)

    def frame(n):
        return pd.DataFrame(
            {
                "few_values": rng.integers(0, 3, n).astype(float),
                "constant": np.zeros(n),
                "continuous": rng.random(n),
            }
        )

    fingerprint = compute_fingerprint(frame(200), frame(4000), columns=list(frame(1).columns))

    assert set(fingerprint["column_drift_scores"]) == {"few_values", "constant", "continuous"}
    assert fingerprint["drift_share"] is not None
