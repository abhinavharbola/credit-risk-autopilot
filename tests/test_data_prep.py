import numpy as np
import pandas as pd

from src.data.split import (
    apply_imputation,
    build_pretrain_batches,
    carve_holdout,
    carve_stream,
    fit_imputation_medians,
)
from src.model.features import TARGET


def make_synthetic_df(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            TARGET: rng.choice([0, 1], size=n, p=[0.933, 0.067]),
            "RevolvingUtilizationOfUnsecuredLines": rng.random(n),
            "age": rng.integers(21, 90, size=n),
            "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 5, size=n),
            "DebtRatio": rng.random(n),
            "MonthlyIncome": rng.choice(
                [np.nan] + list(rng.integers(1000, 10000, size=20)), size=n
            ),
            "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 20, size=n),
            "NumberOfTimes90DaysLate": rng.integers(0, 3, size=n),
            "NumberRealEstateLoansOrLines": rng.integers(0, 5, size=n),
            "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 3, size=n),
            "NumberOfDependents": rng.choice([np.nan, 0, 1, 2, 3], size=n),
        }
    )
    return df


def test_carve_holdout_disjoint_and_covers_all_rows():
    df = make_synthetic_df()
    df["_row_id"] = range(len(df))
    train_pool, holdout = carve_holdout(df, holdout_frac=0.15, seed=1)

    assert len(train_pool) + len(holdout) == len(df)
    assert set(train_pool["_row_id"]).isdisjoint(set(holdout["_row_id"]))
    assert len(holdout) > 0
    assert len(train_pool) > 0


def test_carve_holdout_stratifies_positive_class():
    df = make_synthetic_df(n=5000)
    train_pool, holdout = carve_holdout(df, holdout_frac=0.15, seed=1)

    train_rate = train_pool[TARGET].mean()
    holdout_rate = holdout[TARGET].mean()
    assert abs(train_rate - holdout_rate) < 0.03


def test_carve_holdout_is_deterministic_given_seed():
    df = make_synthetic_df()
    train_a, holdout_a = carve_holdout(df, seed=7)
    train_b, holdout_b = carve_holdout(df, seed=7)

    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(holdout_a, holdout_b)


def test_medians_fit_on_train_pool_only_not_holdout():
    df = make_synthetic_df(n=200, seed=3)
    train_pool, holdout = carve_holdout(df, holdout_frac=0.2, seed=3)

    holdout = holdout.copy()
    holdout["MonthlyIncome"] = 999_999.0

    medians_from_train_only = fit_imputation_medians(train_pool)
    medians_if_leaked = fit_imputation_medians(pd.concat([train_pool, holdout]))

    assert medians_from_train_only["MonthlyIncome"] != medians_if_leaked["MonthlyIncome"]
    assert medians_from_train_only["MonthlyIncome"] < 999_999.0


def test_apply_imputation_fills_only_target_columns():
    df = make_synthetic_df(n=50, seed=5)
    medians = fit_imputation_medians(df)
    imputed = apply_imputation(df, medians)

    assert imputed["MonthlyIncome"].isna().sum() == 0
    assert imputed["NumberOfDependents"].isna().sum() == 0
    pd.testing.assert_series_equal(imputed["age"], df["age"])


def test_build_pretrain_batches_fixed_size_and_reproducible():
    df = make_synthetic_df(n=500, seed=9)
    medians = fit_imputation_medians(df)
    imputed = apply_imputation(df, medians)

    batches_a = build_pretrain_batches(imputed, batch_size=50, seed=11)
    batches_b = build_pretrain_batches(imputed, batch_size=50, seed=11)

    assert len(batches_a) == 10
    assert all(len(b) == 50 for b in batches_a)
    for a, b in zip(batches_a, batches_b):
        pd.testing.assert_frame_equal(a, b)


def test_stream_batches_never_overlap_the_base_training_pool():
    df = make_synthetic_df(n=6000)
    df["_row_id"] = range(len(df))
    remainder, holdout = carve_holdout(df, holdout_frac=0.15, seed=1)
    base_pool, stream_pool = carve_stream(remainder, stream_frac=0.5, seed=1)
    batches = build_pretrain_batches(stream_pool, batch_size=100, seed=1)

    batch_ids = set().union(*(set(b["_row_id"]) for b in batches))
    assert batch_ids.isdisjoint(set(base_pool["_row_id"]))
    assert batch_ids.isdisjoint(set(holdout["_row_id"]))
    assert set(base_pool["_row_id"]).isdisjoint(set(holdout["_row_id"]))


def test_carve_stream_covers_the_remainder_and_stratifies():
    df = make_synthetic_df(n=8000)
    df["_row_id"] = range(len(df))
    base_pool, stream_pool = carve_stream(df, stream_frac=0.5, seed=3)

    assert len(base_pool) + len(stream_pool) == len(df)
    assert abs(len(stream_pool) - len(df) / 2) < 5
    assert abs(base_pool[TARGET].mean() - stream_pool[TARGET].mean()) < 0.02
