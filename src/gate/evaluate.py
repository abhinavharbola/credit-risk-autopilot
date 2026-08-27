"""Pure gate logic. Zero I/O, most heavily tested file in the repo (build
order step 4). Three gates, all must pass to promote a challenger:

  1. tolerance-band  - challenger is not meaningfully worse than champion.
  2. dominance       - challenger genuinely beats champion, not just a tie
                        within the tolerance band.
  3. significance    - that improvement is statistically distinguishable
                        from noise: McNemar's test on matched champion/
                        challenger predictions over the identical batch, or
                        a bootstrap CI on the metric delta when there are
                        too few discordant pairs to trust McNemar.

Primary metric is AUC-PR (documented in config/gate_config.yaml), not
accuracy, since accuracy is close to useless at ~6.7% positive rate.
"""

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
    significance_stat: float | None
    significance_pvalue: float | None
    passed_significance: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form for audit_log.event_payload."""
        return asdict(self)


def compute_metric(y_true, y_prob, metric: str, decision_threshold: float) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if metric == "auc_pr":
        if y_true.sum() == 0:
            # no positives in this slice, AUC-PR is undefined; treat as worst case
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
    """Paired McNemar test with continuity correction on the classification
    (correct/incorrect vs y_true) agreement between champion and challenger.
    Returns (chi2_stat, pvalue, n_discordant_pairs).
    """
    b = int(np.sum(champion_correct & ~challenger_correct))
    c = int(np.sum(~champion_correct & challenger_correct))
    n_discordant = b + c
    if n_discordant == 0:
        return 0.0, 1.0, 0
    chi2 = (abs(b - c) - 1) ** 2 / n_discordant
    pvalue = float(1 - stats.chi2.cdf(chi2, df=1))
    return float(chi2), pvalue, n_discordant


def bootstrap_metric_ci(
    y_true,
    y_prob,
    metric: str,
    decision_threshold: float,
    n_resamples: int,
    seed: int,
    alpha: float,
) -> tuple[float, float]:
    """Bootstrap CI on a single model's metric over one batch, resampling
    rows with replacement. Public (unlike _bootstrap_delta_ci above, which
    is champion-vs-challenger specific): used by the rollback check
    (src/orchestration/promote.py) to confirm an apparent metric drop holds
    up across resamples of the batch, not just a single noisy point
    estimate from ~200 rows and a handful of positives - the same class of
    small-sample risk the gate's own significance check already guards
    against on the promotion side. Before this, rollback triggered on a
    raw point-estimate threshold with no equivalent safeguard.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    values = np.empty(n_resamples)

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        values[i] = compute_metric(y_true[idx], y_prob[idx], metric, decision_threshold)

    lower = float(np.percentile(values, 100 * (alpha / 2)))
    upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return lower, upper


def _bootstrap_delta_ci(
    y_true: np.ndarray,
    champion_prob: np.ndarray,
    challenger_prob: np.ndarray,
    metric: str,
    decision_threshold: float,
    n_resamples: int,
    seed: int,
    alpha: float,
) -> tuple[float, float]:
    """Bootstrap CI on (challenger_metric - champion_metric), resampling
    matched rows with replacement so champion and challenger always see the
    exact same resampled indices in each iteration.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_resamples)

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        champ_m = compute_metric(y_true[idx], champion_prob[idx], metric, decision_threshold)
        chal_m = compute_metric(y_true[idx], challenger_prob[idx], metric, decision_threshold)
        deltas[i] = chal_m - champ_m

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
    """Evaluates whether challenger should be promoted over champion.

    champion_prob, challenger_prob, y_true must all be the same length and
    the same row order: both models scored on the identical newly-labeled
    batch. Enforcing that alignment is the caller's job (see
    src/orchestration/pipeline.py) - this function only validates lengths
    match, it cannot detect a mismatched-window bug on its own.
    """
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
            "(matched batch) - mismatched lengths usually mean a stale "
            "prediction table or the wrong window was passed in"
        )

    champion_metric = compute_metric(y_true, champion_prob, metric, threshold)
    challenger_metric = compute_metric(y_true, challenger_prob, metric, threshold)
    delta = challenger_metric - champion_metric

    passed_tolerance = challenger_metric >= champion_metric - tolerance_band
    passed_dominance = challenger_metric > champion_metric

    champion_pred = (champion_prob >= threshold).astype(int)
    challenger_pred = (challenger_prob >= threshold).astype(int)
    champion_correct = champion_pred == y_true
    challenger_correct = challenger_pred == y_true

    chi2, pvalue, n_discordant = _mcnemar(champion_correct, challenger_correct)
    details: dict[str, Any] = {"n_discordant_pairs": n_discordant, "n_samples": len(y_true)}

    if n_discordant >= min_discordant:
        significance_method = "mcnemar"
        significance_stat = chi2
        significance_pvalue = pvalue
        passed_significance = passed_dominance and pvalue < alpha
    else:
        significance_method = "bootstrap"
        lower, upper = _bootstrap_delta_ci(
            y_true, champion_prob, challenger_prob, metric, threshold,
            n_resamples, seed, alpha,
        )
        significance_stat = None
        significance_pvalue = None
        passed_significance = lower > 0
        details["bootstrap_ci_lower"] = lower
        details["bootstrap_ci_upper"] = upper

    promote = passed_tolerance and passed_dominance and passed_significance

    if not passed_tolerance:
        reason = "challenger metric falls below champion beyond tolerance band, rejected"
    elif not passed_dominance:
        reason = "challenger does not exceed champion (tie or worse within tolerance), rejected"
    elif not passed_significance:
        reason = (
            f"improvement not statistically significant via {significance_method}, "
            "likely noise, rejected"
        )
    else:
        reason = f"challenger significantly beats champion via {significance_method}, promoted"

    return GateResult(
        promote=promote,
        primary_metric=metric,
        champion_metric=champion_metric,
        challenger_metric=challenger_metric,
        delta=delta,
        tolerance_band=tolerance_band,
        passed_tolerance=passed_tolerance,
        passed_dominance=passed_dominance,
        significance_method=significance_method,
        significance_stat=significance_stat,
        significance_pvalue=significance_pvalue,
        passed_significance=passed_significance,
        reason=reason,
        details=details,
    )
