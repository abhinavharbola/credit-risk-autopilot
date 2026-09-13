"""Shared display formatting for dashboard views. Metric names come straight
from src.gate.evaluate's raw identifiers (auc_pr, recall_at_threshold, ...):
fine as dict keys, wrong as UI copy. str.title() alone still gets acronyms
wrong ("Auc pr" instead of "AUC-PR"), so known metrics get an explicit label
and anything unrecognized falls back to a reasonable title-case guess rather
than crashing on a metric this dict hasn't been updated for yet.
"""

METRIC_LABELS = {
    "auc_pr": "AUC-PR",
    "recall_at_threshold": "Recall @ threshold",
    "precision_at_threshold": "Precision @ threshold",
}


def metric_label(name: str) -> str:
    return METRIC_LABELS.get(name, name.replace("_", " ").title())