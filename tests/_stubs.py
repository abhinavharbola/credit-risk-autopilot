"""Shared dependency stubs for offline testing. This sandbox has no network,
so mlflow/evidently/sqlalchemy/dotenv aren't installable here. Every test
file that needs to import a src module touching these packages should
`import tests._stubs` as its very first import, before any src import.

Single source of truth on purpose: multiple test files each defining their
own partial stub and racing to register first (via sys.modules.setdefault)
is exactly the kind of duplicated-logic bug this project's own review
flagged elsewhere - whichever file imports first would "win" with an
incomplete stub, and every other file would silently inherit it instead of
getting its own complete one.

In a real environment with these packages actually installed, every _stub()
call below is a no-op.
"""

import sys
import types
from unittest.mock import MagicMock


def _stub(name: str, **attrs) -> None:
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except ImportError:
        pass
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod


_stub("evidently", Report=object)
_stub("evidently.presets", DataDriftPreset=object)
_stub("mlflow.sklearn", load_model=lambda uri: f"model_for_{uri}")
_stub("mlflow.pyfunc", load_model=lambda uri: f"model_for_{uri}")
_stub(
    "mlflow",
    MlflowClient=lambda *a, **k: MagicMock(),
    start_run=MagicMock(),
    sklearn=sys.modules.get("mlflow.sklearn"),
    pyfunc=sys.modules.get("mlflow.pyfunc"),
    log_param=MagicMock(),
)
_stub("sqlalchemy", text=lambda s: s)
_stub("sqlalchemy.engine", Connection=object)
_stub("dotenv", load_dotenv=lambda *a, **k: None)
