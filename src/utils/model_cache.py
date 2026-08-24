"""Shared model-loading cache: reloads a registered model's pyfunc artifact
only when the given alias's version has actually changed. Used by both the
serving layer (src/serving/app.py) and the orchestration loop
(src/orchestration/pipeline.py), so there's exactly one cache
implementation, not two independently-written copies that could drift apart.
"""

from typing import Any

import mlflow


class AliasedModelCache:
    """get() is safe to call as often as needed (once per request, once per
    tick): the alias-version metadata lookup is cheap, the actual model
    artifact download only happens when the alias has moved since the last
    call.
    """

    def __init__(self, model_name: str, alias: str):
        self.model_name = model_name
        self.alias = alias
        self._version: str | None = None
        self._model = None

    def get(self) -> tuple[Any, str]:
        client = mlflow.MlflowClient()
        try:
            version_info = client.get_model_version_by_alias(self.model_name, self.alias)
        except Exception as e:
            raise RuntimeError(
                f"no model is currently aliased @{self.alias} for {self.model_name}"
            ) from e

        if version_info.version != self._version:
            self._model = mlflow.pyfunc.load_model(
                f"models:/{self.model_name}@{self.alias}"
            )
            self._version = version_info.version

        return self._model, self._version
