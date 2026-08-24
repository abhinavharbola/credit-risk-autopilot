"""Thin YAML config loader. Keeps drift params and gate config as versioned
files, never inline magic numbers in the modules that use them.
"""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)
