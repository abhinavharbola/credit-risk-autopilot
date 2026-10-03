import numpy as np
import pytest

from src.gate.evaluate import bootstrap_delta_ci, compute_metric, evaluate_gate

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
    challenger_prob = rng.uniform(0.0, 1.0, n)

    result = evaluate_gate(y_true, champion_prob, challenger_prob, CONFIG)

    assert result.passed_tolerance is False
    assert result.promote is False
    assert "tolerance band" in result.reason


def test_rejects_when_challenger_ties_champion_within_tolerance():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    champion_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8])
    challenger_prob = champion_prob.copy()

    result = evaluate_gate(y_true, champion_prob, challenger_prob, CONFIG)

    assert result.delta == 0
    assert result.passed_tolerance is True
    assert result.passed_dominance is False
    assert result.promote is False
    assert "does not exceed" in result.reason


def test_rejects_challenger_that_looks_better_only_due_to_noisy_small_sample():
    rng = np.random.default_rng(0)
    n = 15
    y_true = rng.choice([0, 1], size=n, p=[0.8, 0.2])
    champion_prob = rng.uniform(0, 1, size=n)
    challenger_prob = np.clip(champion_prob + rng.normal(0, 0.15, size=n), 0, 1)

    result = evaluate_gate(y_true, champion_prob, challenger_prob, CONFIG, seed=0)

    assert result.significance_method == "bootstrap"
    assert result.passed_dominance is True
    assert result.passed_significance is False
    assert result.promote is False


def test_promotes_when_challenger_is_clearly_and_significantly_better():
    rng = np.random.default_rng(2)
    n = 400
    y_true = rng.choice([0, 1], size=n, p=[0.9, 0.1])
    champion_prob = np.where(
        y_true == 1, rng.uniform(0.3, 0.6, n), rng.uniform(0.2, 0.5, n)
    )
    challenger_prob = np.where(
        y_true == 1, rng.uniform(0.7, 1.0, n), rng.uniform(0.0, 0.3, n)
    )

    result = evaluate_gate(y_true, champion_prob, challenger_prob, CONFIG)

    assert result.passed_tolerance is True
    assert result.passed_dominance is True
    assert result.passed_significance is True
    assert result.promote is True
    assert result.significance_method == "bootstrap"
    assert result.details["bootstrap_ci_lower"] > 0
    assert result.details["mcnemar_reliable"] is True
    assert result.details["mcnemar_pvalue"] < CONFIG["significance_alpha"]


def test_mcnemar_diagnostic_reported_when_discordant_pairs_meet_minimum():
    rng = np.random.default_rng(3)
    n = 200
    y_true = rng.choice([0, 1], size=n, p=[0.85, 0.15])
    champion_prob = rng.uniform(0, 1, n)
    challenger_prob = np.where(y_true == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))

    result = evaluate_gate(y_true, champion_prob, challenger_prob, CONFIG)

    assert result.details["mcnemar_n_discordant_pairs"] >= CONFIG["mcnemar_min_discordant_pairs"]
    assert result.details["mcnemar_reliable"] is True
    assert result.significance_method == "bootstrap"


def test_mcnemar_diagnostic_flagged_unreliable_below_min_discordant_pairs():
    y_true = np.array([0, 1, 0, 1, 0])
    champion_prob = np.array([0.1, 0.9, 0.2, 0.3, 0.1])
    challenger_prob = np.array([0.1, 0.95, 0.2, 0.85, 0.1])

    result = evaluate_gate(y_true, champion_prob, challenger_prob, CONFIG)

    assert result.details["mcnemar_n_discordant_pairs"] < CONFIG["mcnemar_min_discordant_pairs"]
    assert result.details["mcnemar_reliable"] is False
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


def test_bootstrap_delta_ci_returns_lower_le_upper():
    rng = np.random.default_rng(3)
    y_true = rng.choice([0, 1], size=300, p=[0.9, 0.1])
    baseline = rng.uniform(0, 1, 300)
    candidate = np.clip(baseline + rng.normal(0, 0.1, 300), 0, 1)

    lower, upper = bootstrap_delta_ci(y_true, baseline, candidate, "auc_pr", 0.5, 300, 42, 0.05)

    assert lower <= upper


def test_bootstrap_delta_ci_is_reproducible_given_same_seed():
    rng = np.random.default_rng(4)
    y_true = rng.choice([0, 1], size=300, p=[0.9, 0.1])
    baseline = rng.uniform(0, 1, 300)
    candidate = rng.uniform(0, 1, 300)

    ci_a = bootstrap_delta_ci(y_true, baseline, candidate, "auc_pr", 0.5, 300, 42, 0.05)
    ci_b = bootstrap_delta_ci(y_true, baseline, candidate, "auc_pr", 0.5, 300, 42, 0.05)

    assert ci_a == ci_b


def test_bootstrap_delta_ci_is_symmetric_under_swapping_roles():
    rng = np.random.default_rng(5)
    y_true = rng.choice([0, 1], size=400, p=[0.85, 0.15])
    a = np.where(y_true == 1, rng.uniform(0.4, 1.0, 400), rng.uniform(0.0, 0.7, 400))
    b = rng.uniform(0, 1, 400)

    lo_ab, hi_ab = bootstrap_delta_ci(y_true, a, b, "auc_pr", 0.5, 400, 7, 0.05)
    lo_ba, hi_ba = bootstrap_delta_ci(y_true, b, a, "auc_pr", 0.5, 400, 7, 0.05)

    assert lo_ab == pytest.approx(-hi_ba)
    assert hi_ab == pytest.approx(-lo_ba)


def test_promotion_is_equivalent_to_the_significance_gate():
    rng = np.random.default_rng(6)
    for seed in range(12):
        n = int(rng.integers(60, 250))
        y_true = rng.choice([0, 1], size=n, p=[0.88, 0.12])
        champion_prob = rng.uniform(0, 1, n)
        challenger_prob = np.clip(champion_prob + rng.normal(0.02, 0.2, n), 0, 1)

        result = evaluate_gate(y_true, champion_prob, challenger_prob, CONFIG, seed=seed)

        assert result.promote == result.passed_significance
        if result.passed_significance:
            assert result.passed_dominance and result.passed_tolerance
