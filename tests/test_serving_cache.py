"""Tests AliasedModelCache (src/utils/model_cache.py) - the single cache
implementation shared by the serving layer and the orchestration loop, so a
fix here covers both call sites at once instead of two independently-tested
copies.
"""

import tests._stubs  # noqa: F401  (must run before src imports below)

from unittest.mock import MagicMock

from src.utils.model_cache import AliasedModelCache


def test_cache_reloads_only_when_alias_version_changes():
    cache = AliasedModelCache("credit-risk-classifier", "production")

    fake_client = MagicMock()
    fake_client.get_model_version_by_alias.return_value = MagicMock(version="1")

    import src.utils.model_cache as model_cache_module

    orig_client_cls = model_cache_module.mlflow.MlflowClient
    orig_loader = model_cache_module.mlflow.pyfunc.load_model
    model_cache_module.mlflow.MlflowClient = lambda: fake_client
    load_calls = []
    model_cache_module.mlflow.pyfunc.load_model = lambda uri: load_calls.append(uri) or f"model_for_{uri}"

    try:
        model_a, version_a = cache.get()
        model_b, version_b = cache.get()
    finally:
        model_cache_module.mlflow.MlflowClient = orig_client_cls
        model_cache_module.mlflow.pyfunc.load_model = orig_loader

    assert version_a == version_b == "1"
    assert model_a == model_b
    assert len(load_calls) == 1  # second call did not reload


def test_cache_reloads_after_alias_version_bumps():
    cache = AliasedModelCache("credit-risk-classifier", "production")

    fake_client = MagicMock()
    fake_client.get_model_version_by_alias.side_effect = [
        MagicMock(version="1"),
        MagicMock(version="2"),  # promotion happened between calls
    ]

    import src.utils.model_cache as model_cache_module

    orig_client_cls = model_cache_module.mlflow.MlflowClient
    orig_loader = model_cache_module.mlflow.pyfunc.load_model
    model_cache_module.mlflow.MlflowClient = lambda: fake_client
    load_calls = []
    model_cache_module.mlflow.pyfunc.load_model = lambda uri: load_calls.append(uri) or f"model_for_{uri}"

    try:
        _, version_a = cache.get()
        _, version_b = cache.get()
    finally:
        model_cache_module.mlflow.MlflowClient = orig_client_cls
        model_cache_module.mlflow.pyfunc.load_model = orig_loader

    assert version_a == "1"
    assert version_b == "2"
    assert len(load_calls) == 2  # reloaded exactly once, on the version change


def test_cache_raises_runtime_error_when_no_alias_set():
    cache = AliasedModelCache("credit-risk-classifier", "production")

    fake_client = MagicMock()
    fake_client.get_model_version_by_alias.side_effect = Exception("alias not found")

    import src.utils.model_cache as model_cache_module

    orig_client_cls = model_cache_module.mlflow.MlflowClient
    model_cache_module.mlflow.MlflowClient = lambda: fake_client

    try:
        try:
            cache.get()
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "no model is currently aliased" in str(e)
    finally:
        model_cache_module.mlflow.MlflowClient = orig_client_cls
