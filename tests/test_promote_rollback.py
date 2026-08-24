"""Tests for the pure logic in promote.py and detect.py's staleness check.
mlflow/evidently/sqlalchemy aren't installed in this sandbox (no network);
tests._stubs provides a single, consistent set of stubs for their import
surface. In a real environment with those packages installed, the stubs are
a no-op and tests run against the real imports.
"""

import tests._stubs  # noqa: F401  (must run before src imports below)

from src.drift.detect import check_fingerprint_staleness
from src.orchestration.promote import find_previous_champion


def test_find_previous_champion_skips_rolled_back_and_future_entries():
    history = [
        {"id": 1, "rolled_back_at": None},
        {"id": 2, "rolled_back_at": "2026-01-01"},  # was rolled back, not a valid target
        {"id": 3, "rolled_back_at": None},
        {"id": 4, "rolled_back_at": None},  # current champion, excluded by caller
    ]
    previous = find_previous_champion(history, current_champion_history_id=4)
    assert previous["id"] == 3


def test_find_previous_champion_skips_to_older_hop_if_immediate_predecessor_was_rolled_back():
    history = [
        {"id": 1, "rolled_back_at": None},
        {"id": 2, "rolled_back_at": "2026-01-01"},
        {"id": 3, "rolled_back_at": "2026-01-02"},
    ]
    previous = find_previous_champion(history, current_champion_history_id=3)
    assert previous["id"] == 1


def test_find_previous_champion_returns_none_when_no_valid_candidate():
    history = [{"id": 1, "rolled_back_at": None}]
    previous = find_previous_champion(history, current_champion_history_id=1)
    assert previous is None


def test_fingerprint_staleness_flags_large_drift_share_delta():
    then = {"drift_share": 0.1, "column_drift_scores": {}}
    now = {"drift_share": 0.5, "column_drift_scores": {}}
    assert check_fingerprint_staleness(then, now, threshold=0.1) is True


def test_fingerprint_staleness_false_within_threshold():
    then = {"drift_share": 0.2, "column_drift_scores": {"DebtRatio": 0.3}}
    now = {"drift_share": 0.22, "column_drift_scores": {"DebtRatio": 0.32}}
    assert check_fingerprint_staleness(then, now, threshold=0.1) is False


def test_fingerprint_staleness_flags_large_column_score_delta_even_if_share_stable():
    then = {"drift_share": 0.2, "column_drift_scores": {"DebtRatio": 0.1}}
    now = {"drift_share": 0.21, "column_drift_scores": {"DebtRatio": 0.8}}
    assert check_fingerprint_staleness(then, now, threshold=0.1) is True


def test_fingerprint_staleness_handles_no_shared_columns():
    then = {"drift_share": 0.1, "column_drift_scores": {"DebtRatio": 0.1}}
    now = {"drift_share": 0.11, "column_drift_scores": {"MonthlyIncome": 0.9}}
    # no shared columns to compare, falls back to drift_share only, which is within threshold
    assert check_fingerprint_staleness(then, now, threshold=0.1) is False
