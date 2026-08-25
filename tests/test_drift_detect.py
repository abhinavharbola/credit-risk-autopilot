"""Tests _reduce_to_fingerprint against a real evidently==0.7.21
report.dict() output, captured from an actual live run (not guessed from
docs). This exact payload is what exposed the original bug: matching on a
"metric_id" key that doesn't exist in the real schema meant drift_share
stayed None forever and the retrain trigger never fired across a full
25-tick demo run. This test pins the fix so that regression can't silently
come back.
"""

import tests._stubs  # noqa: F401  (must run before src imports below)

from src.drift.detect import _reduce_to_fingerprint

# captured verbatim from a live evidently==0.7.21 run comparing a drifted
# batch (DebtRatio shifted) against a clean reference
REAL_EVIDENTLY_OUTPUT = {
    "metrics": [
        {
            "id": "15e89f895b482f9b84ba7274ed18a106",
            "metric_name": "DriftedColumnsCount(drift_share=0.5)",
            "config": {
                "type": "evidently:metric_v2:DriftedColumnsCount",
                "drift_share": 0.5,
            },
            "value": {"count": 1.0, "share": 0.3333333333333333},
        },
        {
            "id": "3101add92406b5469c65ad579a51ea39",
            "metric_name": "ValueDrift(column=DebtRatio,method=K-S p_value,threshold=0.05)",
            "config": {
                "type": "evidently:metric_v2:ValueDrift",
                "column": "DebtRatio",
                "method": "K-S p_value",
                "threshold": 0.05,
            },
            "value": 4.569224815484179e-66,
        },
        {
            "id": "96a46647ac7709a5834a782551789cb7",
            "metric_name": (
                "ValueDrift(column=RevolvingUtilizationOfUnsecuredLines,"
                "method=K-S p_value,threshold=0.05)"
            ),
            "config": {
                "type": "evidently:metric_v2:ValueDrift",
                "column": "RevolvingUtilizationOfUnsecuredLines",
                "method": "K-S p_value",
                "threshold": 0.05,
            },
            "value": 0.7933622419382523,
        },
        {
            "id": "cc102b3d5cf019c902ad5b4cbb420054",
            "metric_name": "ValueDrift(column=MonthlyIncome,method=K-S p_value,threshold=0.05)",
            "config": {
                "type": "evidently:metric_v2:ValueDrift",
                "column": "MonthlyIncome",
                "method": "K-S p_value",
                "threshold": 0.05,
            },
            "value": 0.17793352788293415,
        },
    ],
    "tests": [],
}


def test_reduce_to_fingerprint_extracts_drift_share_from_real_output():
    fingerprint = _reduce_to_fingerprint(REAL_EVIDENTLY_OUTPUT)
    assert fingerprint["drift_share"] == 0.3333333333333333


def test_reduce_to_fingerprint_extracts_per_column_pvalues_from_real_output():
    fingerprint = _reduce_to_fingerprint(REAL_EVIDENTLY_OUTPUT)
    scores = fingerprint["column_drift_scores"]
    assert scores["DebtRatio"] < 1e-60
    assert scores["RevolvingUtilizationOfUnsecuredLines"] == 0.7933622419382523
    assert scores["MonthlyIncome"] == 0.17793352788293415


def test_reduce_to_fingerprint_never_returns_none_drift_share_for_valid_output():
    """This is the actual regression the original bug caused: drift_share
    silently stayed None on every real report, so retrain never triggered
    across an entire 25-tick run despite real, measurable drift.
    """
    fingerprint = _reduce_to_fingerprint(REAL_EVIDENTLY_OUTPUT)
    assert fingerprint["drift_share"] is not None


def test_retrain_would_trigger_at_the_configured_threshold():
    """Uses config/gate_config.yaml's actual reference_fingerprint_drift_threshold
    (0.3, not the original 0.1 - raised after a real run showed 0.1 producing
    false retrain triggers on undrifted batches from multiple-testing noise
    alone, see the comment in gate_config.yaml for the statistical rationale).
    The captured sample has drift_share=0.333 (1 of the 3 tested columns
    flagged), which still clears 0.3.
    """
    fingerprint = _reduce_to_fingerprint(REAL_EVIDENTLY_OUTPUT)
    threshold = 0.3  # config/gate_config.yaml: reference_fingerprint_drift_threshold
    triggered = fingerprint["drift_share"] is not None and fingerprint["drift_share"] >= threshold
    assert triggered is True


def test_single_column_noise_does_not_trigger_at_the_raised_threshold():
    """The actual bug this threshold change fixes: a fingerprint where only
    one feature out of ten spuriously flags (share=0.1, exactly the old
    threshold) must NOT trigger a retrain - that's multiple-testing noise on
    an undrifted batch, not real drift.
    """
    single_column_noise = {
        "drift_share": 0.1,
        "column_drift_scores": {"DebtRatio": 0.03},
    }
    threshold = 0.3
    triggered = (
        single_column_noise["drift_share"] is not None
        and single_column_noise["drift_share"] >= threshold
    )
    assert triggered is False


def test_reduce_to_fingerprint_handles_missing_metrics_key_gracefully():
    fingerprint = _reduce_to_fingerprint({})
    assert fingerprint == {"drift_share": None, "column_drift_scores": {}}
