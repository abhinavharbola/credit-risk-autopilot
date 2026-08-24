"""Unit tests for drift injection: persistent drift activation, temporary
concept drift window, and seed-based reproducibility.
"""

import numpy as np
import pandas as pd

from src.data.drift_injection import (
    apply_persistent_drift,
    apply_temporary_concept_drift,
    inject_drift,
)
from src.model.features import TARGET

PARAMS = {
    "seed": 42,
    "persistent_drift": {
        "start_batch": 10,
        "columns": {
            "DebtRatio": {"shift": 0.15, "scale": 1.20},
            "RevolvingUtilizationOfUnsecuredLines": {"shift": 0.10, "scale": 1.15},
            "MonthlyIncome": {"shift": -0.20, "scale": 1.0},
        },
    },
    "temporary_concept_drift": {
        "start_batch": 15,
        "end_batch": 20,
        "blend_ratio": 0.5,
        "columns": ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"],
    },
}


def make_batch(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            TARGET: rng.choice([0, 1], size=n, p=[0.9, 0.1]),
            "DebtRatio": rng.random(n),
            "RevolvingUtilizationOfUnsecuredLines": rng.random(n),
            "MonthlyIncome": rng.uniform(1000, 8000, size=n),
        }
    )


def test_persistent_drift_inactive_before_start_batch():
    batch = make_batch()
    out = apply_persistent_drift(batch, batch_index=5, params=PARAMS)
    pd.testing.assert_frame_equal(out, batch)


def test_persistent_drift_active_from_start_batch_onward():
    batch = make_batch()
    out = apply_persistent_drift(batch, batch_index=10, params=PARAMS)

    expected_debt_ratio = (batch["DebtRatio"] + 0.15) * 1.20
    pd.testing.assert_series_equal(out["DebtRatio"], expected_debt_ratio)

    expected_income = batch["MonthlyIncome"] * (1 - 0.20) * 1.0
    pd.testing.assert_series_equal(out["MonthlyIncome"], expected_income)


def test_persistent_drift_never_turns_off():
    batch = make_batch()
    out_at_start = apply_persistent_drift(batch, batch_index=10, params=PARAMS)
    out_much_later = apply_persistent_drift(batch, batch_index=500, params=PARAMS)
    pd.testing.assert_frame_equal(out_at_start, out_much_later)


def test_temporary_concept_drift_inactive_outside_window():
    batch = make_batch()
    before = apply_temporary_concept_drift(batch, 14, PARAMS, seed=42)
    after = apply_temporary_concept_drift(batch, 20, PARAMS, seed=42)
    pd.testing.assert_frame_equal(before, batch)
    pd.testing.assert_frame_equal(after, batch)


def test_temporary_concept_drift_active_inside_window_blends_toward_centroid():
    batch = make_batch(n=200, seed=1)
    out = apply_temporary_concept_drift(batch, 17, PARAMS, seed=42)

    non_delinquent_centroid = batch.loc[batch[TARGET] == 0, "DebtRatio"].mean()
    delinquent_before = batch.loc[batch[TARGET] == 1, "DebtRatio"]
    delinquent_after = out.loc[out[TARGET] == 1, "DebtRatio"]

    # blended values must move strictly closer to the centroid than the originals
    dist_before = (delinquent_before - non_delinquent_centroid).abs()
    dist_after = (delinquent_after - non_delinquent_centroid).abs()
    assert (dist_after <= dist_before).all()

    # non-delinquent rows are untouched
    pd.testing.assert_series_equal(
        out.loc[out[TARGET] == 0, "DebtRatio"], batch.loc[batch[TARGET] == 0, "DebtRatio"]
    )


def test_temporary_concept_drift_reverts_after_window_ends():
    batch = make_batch(n=200, seed=1)
    inside_window = apply_temporary_concept_drift(batch, 17, PARAMS, seed=42)
    after_window = apply_temporary_concept_drift(batch, 21, PARAMS, seed=42)

    assert not inside_window.equals(batch)
    pd.testing.assert_frame_equal(after_window, batch)


def test_inject_drift_is_reproducible_given_same_seed_and_batch_index():
    batch = make_batch(n=150, seed=2)
    out_a = inject_drift(batch, batch_index=17, params=PARAMS)
    out_b = inject_drift(batch, batch_index=17, params=PARAMS)
    pd.testing.assert_frame_equal(out_a, out_b)


def test_inject_drift_combines_persistent_and_temporary_in_overlap_window():
    """Batch 17 is inside both the persistent-drift range (>=10) and the
    temporary window (15-20): both effects should be present.
    """
    batch = make_batch(n=150, seed=2)
    out = inject_drift(batch, batch_index=17, params=PARAMS)

    persistent_only = apply_persistent_drift(batch, 17, PARAMS)
    assert not out.equals(persistent_only)  # temporary drift added more change
