"""Tests for the pure gate logic. This is the heaviest-tested file in the
repo (build order step 4) - zero I/O, all synthetic inputs.
"""

import numpy as np

from src.gate.evaluate import bootstrap_metric_ci, compute_metric, evaluate_gate

CONFIG = {
    "primary_metric": "auc_pr",
    "decision_threshold": 0.5,
    "tolerance_band": 0.01,
    "significance_alpha": 0.05,
    "mcnemar_min_discordant_pairs": 15,
    "bootstrap_resamples": 2000,
}


def test_rejects_when_challenger_falls_below_tolerance_band():
    rng = np.random.default_rng(1)
    n = 200
    y_true = rng.choice([0, 1], size=n, p=[0.9, 0.1])
    champion_prob = np.where(y_true == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))
    challenger_prob = rng.uniform(0.0, 1.0, n)  # essentially random, much worse

    result = evaluate_gate(y_true, champion_prob, challenger_prob, CONFIG)

    assert result.passed_tolerance is False
    assert result.promote is False
    assert "tolerance band" in result.reason


def test_rejects_when_challenger_ties_champion_within_tolerance():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    champion_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8])
    challenger_prob = champion_prob.copy()  # identical, delta == 0

    result = evaluate_gate(y_true, champion_prob, challenger_prob, CONFIG)

    assert result.delta == 0
    assert result.passed_tolerance is True
    assert result.passed_dominance is False
    assert result.promote is False
    assert "does not exceed" in result.reason


def test_rejects_challenger_that_looks_better_only_due_to_noisy_small_sample():
    """Definition of done: the gate must reject a challenger that appears
    better purely from small-sample noise. Fixed seed, reproducible.
    """
    rng = np.random.default_rng(0)
    n = 15
    y_true = rng.choice([0, 1], size=n, p=[0.8, 0.2])
    champion_prob = rng.uniform(0, 1, size=n)
    challenger_prob = np.clip(champion_prob + rng.normal(0, 0.15, size=n), 0, 1)

    result = evaluate_gate(y_true, champion_prob, challenger_prob, CONFIG, seed=0)

    assert result.significance_method == "bootstrap"  # n=15 batch, too few discordant pairs for McNemar
    assert result.passed_dominance is True  # looks better on point estimate
    assert result.passed_significance is False  # but not distinguishable from noise
    assert result.promote is False


def test_promotes_when_challenger_is_clearly_and_significantly_better():
    rng = np.random.default_rng(2)
    n = 400
    y_true = rng.choice([0, 1], size=n, p=[0.9, 0.1])
    # champion: weak separation between classes
    champion_prob = np.where(
        y_true == 1, rng.uniform(0.3, 0.6, n), rng.uniform(0.2, 0.5, n)
    )
    # challenger: strong, consistent separation
    challenger_prob = np.where(
        y_true == 1, rng.uniform(0.7, 1.0, n), rng.uniform(0.0, 0.3, n)
    )

    result = evaluate_gate(y_true, champion_prob, challenger_prob, CONFIG)

    assert result.passed_tolerance is True
    assert result.passed_dominance is True
    assert result.passed_significance is True
    assert result.promote is True
    assert result.significance_method == "mcnemar"
    assert result.significance_pvalue is not None
    assert result.significance_pvalue < CONFIG["significance_alpha"]


def test_mcnemar_used_when_discordant_pairs_meet_minimum():
    rng = np.random.default_rng(3)
    n = 200
    y_true = rng.choice([0, 1], size=n, p=[0.85, 0.15])
    champion_prob = rng.uniform(0, 1, n)
    challenger_prob = np.where(y_true == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))

    result = evaluate_gate(y_true, champion_prob, challenger_prob, CONFIG)

    assert result.details["n_discordant_pairs"] >= CONFIG["mcnemar_min_discordant_pairs"]
    assert result.significance_method == "mcnemar"


def test_bootstrap_fallback_used_below_min_discordant_pairs():
    y_true = np.array([0, 1, 0, 1, 0])
    champion_prob = np.array([0.1, 0.9, 0.2, 0.3, 0.1])
    challenger_prob = np.array([0.1, 0.95, 0.2, 0.85, 0.1])

    result = evaluate_gate(y_true, champion_prob, challenger_prob, CONFIG)

    assert result.details["n_discordant_pairs"] < CONFIG["mcnemar_min_discordant_pairs"]
    assert result.significance_method == "bootstrap"
    assert "bootstrap_ci_lower" in result.details


def test_mismatched_lengths_raise():
    y_true = np.array([0, 1, 0])
    champion_prob = np.array([0.1, 0.9])
    challenger_prob = np.array([0.1, 0.9, 0.2])

    try:
        evaluate_gate(y_true, champion_prob, challenger_prob, CONFIG)
        raise AssertionError("expected ValueError for mismatched lengths")
    except ValueError as e:
        assert "matched batch" in str(e)


def test_compute_metric_auc_pr_no_positives_returns_zero_not_nan():
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])

    metric = compute_metric(y_true, y_prob, "auc_pr", decision_threshold=0.5)

    assert metric == 0.0


def test_gate_result_to_dict_is_json_serializable():
    import json

    y_true = np.array([0, 1, 0, 1])
    champion_prob = np.array([0.2, 0.8, 0.3, 0.7])
    challenger_prob = np.array([0.1, 0.9, 0.2, 0.8])

    result = evaluate_gate(y_true, champion_prob, challenger_prob, CONFIG)
    serialized = json.dumps(result.to_dict())
    assert isinstance(serialized, str)


def test_bootstrap_metric_ci_returns_lower_le_upper():
    rng = np.random.default_rng(7)
    n = 200
    y_true = rng.choice([0, 1], size=n, p=[0.9, 0.1])
    y_prob = np.where(y_true == 1, rng.uniform(0.5, 1.0, n), rng.uniform(0.0, 0.5, n))

    lower, upper = bootstrap_metric_ci(
        y_true, y_prob, "auc_pr", decision_threshold=0.5,
        n_resamples=500, seed=1, alpha=0.05,
    )

    assert lower <= upper


def test_bootstrap_metric_ci_is_reproducible_given_same_seed():
    rng = np.random.default_rng(3)
    n = 150
    y_true = rng.choice([0, 1], size=n, p=[0.85, 0.15])
    y_prob = rng.uniform(0, 1, n)

    ci_a = bootstrap_metric_ci(y_true, y_prob, "auc_pr", 0.5, 500, 42, 0.05)
    ci_b = bootstrap_metric_ci(y_true, y_prob, "auc_pr", 0.5, 500, 42, 0.05)

    assert ci_a == ci_b


def test_bootstrap_metric_ci_narrows_with_larger_sample():
    """Larger, more consistent samples should produce a tighter CI than a
    small noisy one - a basic sanity check that this behaves like a real CI.
    Uses overlapping (not perfectly separable) score distributions, since
    perfect separation pins AUC-PR at 1.0 regardless of sample size and
    leaves nothing for sample size to narrow.
    """
    rng = np.random.default_rng(11)

    small_y = rng.choice([0, 1], size=30, p=[0.8, 0.2])
    small_prob = np.where(small_y == 1, rng.uniform(0.3, 0.9, 30), rng.uniform(0.1, 0.7, 30))
    small_lower, small_upper = bootstrap_metric_ci(small_y, small_prob, "auc_pr", 0.5, 1000, 1, 0.05)

    large_y = rng.choice([0, 1], size=3000, p=[0.8, 0.2])
    large_prob = np.where(large_y == 1, rng.uniform(0.3, 0.9, 3000), rng.uniform(0.1, 0.7, 3000))
    large_lower, large_upper = bootstrap_metric_ci(large_y, large_prob, "auc_pr", 0.5, 1000, 1, 0.05)

    assert (large_upper - large_lower) < (small_upper - small_lower)
