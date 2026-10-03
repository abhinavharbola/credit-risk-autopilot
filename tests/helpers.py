import numpy as np
import pandas as pd

from src.model.features import TARGET

GATE_CONFIG = {
    "primary_metric": "auc_pr",
    "decision_threshold": 0.5,
    "tolerance_band": 0.01,
    "significance_alpha": 0.05,
    "mcnemar_min_discordant_pairs": 15,
    "bootstrap_resamples": 300,
    "rollback_metric_drop_threshold": 0.03,
    "retrain_drift_share_threshold": 0.3,
    "retrain_cooldown_batches": 3,
    "retrain_base_pool_sample": 40,
    "retrain_max_labeled_rows": 60,
    "drift_ks_pvalue_threshold": 0.05,
    "staleness_drift_share_delta_threshold": 0.3,
    "staleness_drifted_column_disagreement_threshold": 0.3,
}

PIPELINE_CONFIG = {
    "persistent_drift": {"start_batch": 10, "columns": {}},
    "temporary_concept_drift": {
        "start_batch": 15,
        "end_batch": 20,
        "blend_ratio": 0.5,
        "columns": ["DebtRatio"],
    },
    "delayed_labels": {"delay_batches": 3},
    "gate": GATE_CONFIG,
}


def make_labeled_batch(n=100, seed=0, signal=0.0):
    rng = np.random.default_rng(seed)
    utilization = rng.random(n)
    target = (rng.random(n) < 0.08 + signal * utilization).astype(int)
    return pd.DataFrame(
        {
            TARGET: target,
            "RevolvingUtilizationOfUnsecuredLines": utilization,
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
