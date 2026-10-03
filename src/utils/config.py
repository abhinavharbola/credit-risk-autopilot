from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "credit-risk-classifier"
PRODUCTION_ALIAS = "production"


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(resolve_path(path)) as f:
        return yaml.safe_load(f)


def load_pipeline_config() -> dict[str, Any]:
    drift_params = load_yaml("config/drift_params.yaml")
    gate_config = load_yaml("config/gate_config.yaml")
    return {**drift_params, "gate": gate_config}
