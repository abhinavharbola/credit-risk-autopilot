from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, precision_score, recall_score


@dataclass
class GateResult:
    promote: bool
    primary_metric: str
    champion_metric: float
    challenger_metric: float
    delta: float
    tolerance_band: float
    passed_tolerance: bool
    passed_dominance: bool
    significance_method: str
    passed_significance: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metric(y_true, y_prob, metric: str, decision_threshold: float) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if metric == "auc_pr":
        if y_true.sum() == 0:
            return 0.0
        return float(average_precision_score(y_true, y_prob))
    if metric == "recall_at_threshold":
        y_pred = (y_prob >= decision_threshold).astype(int)
        return float(recall_score(y_true, y_pred, zero_division=0))
    if metric == "precision_at_threshold":
        y_pred = (y_prob >= decision_threshold).astype(int)
        return float(precision_score(y_true, y_pred, zero_division=0))
    raise ValueError(f"unknown primary_metric: {metric}")


def _mcnemar(
    champion_correct: np.ndarray, challenger_correct: np.ndarray
) -> tuple[float, float, int]:
    b = int(np.sum(champion_correct & ~challenger_correct))
    c = int(np.sum(~champion_correct & challenger_correct))
    n_discordant = b + c
    if n_discordant == 0:
        return 0.0, 1.0, 0
    chi2 = (abs(b - c) - 1) ** 2 / n_discordant
    pvalue = float(stats.chi2.sf(chi2, df=1))
    return float(chi2), pvalue, n_discordant


def bootstrap_delta_ci(
    y_true,
    baseline_prob,
    candidate_prob,
    metric: str,
    decision_threshold: float,
    n_resamples: int,
    seed: int,
    alpha: float,
) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    baseline_prob = np.asarray(baseline_prob)
    candidate_prob = np.asarray(candidate_prob)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_resamples)

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        baseline_m = compute_metric(y_true[idx], baseline_prob[idx], metric, decision_threshold)
        candidate_m = compute_metric(y_true[idx], candidate_prob[idx], metric, decision_threshold)
        deltas[i] = candidate_m - baseline_m

    lower = float(np.percentile(deltas, 100 * (alpha / 2)))
    upper = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return lower, upper


def evaluate_gate(
    y_true,
    champion_prob,
    challenger_prob,
    config: dict[str, Any],
    seed: int = 42,
) -> GateResult:
    metric = config["primary_metric"]
    threshold = config["decision_threshold"]
    tolerance_band = config["tolerance_band"]
    alpha = config["significance_alpha"]
    min_discordant = config["mcnemar_min_discordant_pairs"]
    n_resamples = config["bootstrap_resamples"]

    y_true = np.asarray(y_true)
    champion_prob = np.asarray(champion_prob)
    challenger_prob = np.asarray(challenger_prob)

    if not (len(y_true) == len(champion_prob) == len(challenger_prob)):
        raise ValueError(
            "y_true, champion_prob, challenger_prob must be the same length "
            "(matched batch): mismatched lengths usually mean a stale "
            "prediction table or the wrong window was passed in"
        )

    champion_metric = compute_metric(y_true, champion_prob, metric, threshold)
    challenger_metric = compute_metric(y_true, challenger_prob, metric, threshold)
    delta = challenger_metric - champion_metric

    passed_tolerance = challenger_metric >= champion_metric - tolerance_band
    passed_dominance = challenger_metric > champion_metric

    champion_correct = (champion_prob >= threshold).astype(int) == y_true
    challenger_correct = (challenger_prob >= threshold).astype(int) == y_true
    mcnemar_chi2, mcnemar_pvalue, n_discordant = _mcnemar(champion_correct, challenger_correct)

    lower, upper = bootstrap_delta_ci(
        y_true, champion_prob, challenger_prob, metric, threshold, n_resamples, seed, alpha
    )
    passed_significance = lower > 0

    details: dict[str, Any] = {
        "n_samples": len(y_true),
        "bootstrap_ci_lower": lower,
        "bootstrap_ci_upper": upper,
        "mcnemar_chi2": mcnemar_chi2,
        "mcnemar_pvalue": mcnemar_pvalue,
        "mcnemar_n_discordant_pairs": n_discordant,
        "mcnemar_reliable": n_discordant >= min_discordant,
    }

    promote = passed_tolerance and passed_dominance and passed_significance

    if not passed_tolerance:
        reason = "challenger metric falls below champion beyond tolerance band, rejected"
    elif not passed_dominance:
        reason = "challenger does not exceed champion (tie or worse within tolerance), rejected"
    elif not passed_significance:
        reason = (
            f"improvement not statistically significant (bootstrap CI on the "
            f"{metric} delta includes zero), likely noise, rejected"
        )
    else:
        reason = (
            f"challenger significantly beats champion (bootstrap CI on the "
            f"{metric} delta), promoted"
        )

    return GateResult(
        promote=promote,
        primary_metric=metric,
        champion_metric=champion_metric,
        challenger_metric=challenger_metric,
        delta=delta,
        tolerance_band=tolerance_band,
        passed_tolerance=passed_tolerance,
        passed_dominance=passed_dominance,
        significance_method="bootstrap",
        passed_significance=passed_significance,
        reason=reason,
        details=details,
    )
