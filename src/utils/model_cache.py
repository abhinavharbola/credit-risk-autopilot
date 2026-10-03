import threading
import time
from typing import Any

import mlflow
import mlflow.sklearn


class AliasedModelCache:
    def __init__(self, model_name: str, alias: str, ttl_seconds: float = 0.0):
        self.model_name = model_name
        self.alias = alias
        self.ttl_seconds = ttl_seconds
        self._version: str | None = None
        self._model = None
        self._checked_at = 0.0
        self._lock = threading.Lock()

    def get(self) -> tuple[Any, str]:
        with self._lock:
            now = time.monotonic()
            if (
                self._model is not None
                and self.ttl_seconds > 0
                and now - self._checked_at < self.ttl_seconds
            ):
                return self._model, self._version

            client = mlflow.MlflowClient()
            try:
                version_info = client.get_model_version_by_alias(self.model_name, self.alias)
            except Exception as e:
                raise RuntimeError(
                    f"no model is currently aliased @{self.alias} for {self.model_name}"
                ) from e

            version = str(version_info.version)
            if version != self._version:
                self._model = mlflow.sklearn.load_model(f"models:/{self.model_name}/{version}")
                self._version = version
            self._checked_at = now
            return self._model, self._version
