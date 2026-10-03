import threading
from unittest.mock import MagicMock

import pytest

import src.utils.model_cache as cache_mod
from src.utils.model_cache import AliasedModelCache


class FakeRegistry:
    def __init__(self, versions):
        self.versions = list(versions)
        self.alias_lookups = 0
        self.loads = []
        self.client = MagicMock()
        self.client.get_model_version_by_alias.side_effect = self.lookup

    def lookup(self, name, alias):
        self.alias_lookups += 1
        version = self.versions[min(self.alias_lookups - 1, len(self.versions) - 1)]
        return MagicMock(version=version)

    def install(self, monkeypatch):
        monkeypatch.setattr(cache_mod.mlflow, "MlflowClient", lambda: self.client)
        monkeypatch.setattr(
            cache_mod.mlflow.sklearn,
            "load_model",
            lambda uri: self.loads.append(uri) or f"model_for_{uri}",
        )


def test_reloads_only_when_alias_version_changes(monkeypatch):
    registry = FakeRegistry(["1", "1"])
    registry.install(monkeypatch)
    cache = AliasedModelCache("m", "production")

    model_a, version_a = cache.get()
    model_b, version_b = cache.get()

    assert version_a == version_b == "1"
    assert model_a == model_b
    assert len(registry.loads) == 1


def test_reloads_after_alias_version_bumps(monkeypatch):
    registry = FakeRegistry(["1", "2"])
    registry.install(monkeypatch)
    cache = AliasedModelCache("m", "production")

    _, version_a = cache.get()
    _, version_b = cache.get()

    assert (version_a, version_b) == ("1", "2")
    assert len(registry.loads) == 2


def test_loads_the_resolved_version_not_the_alias(monkeypatch):
    registry = FakeRegistry(["7"])
    registry.install(monkeypatch)

    AliasedModelCache("m", "production").get()

    assert registry.loads == ["models:/m/7"]


def test_raises_runtime_error_when_no_alias_set(monkeypatch):
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = Exception("alias not found")
    monkeypatch.setattr(cache_mod.mlflow, "MlflowClient", lambda: client)

    with pytest.raises(RuntimeError, match="no model is currently aliased"):
        AliasedModelCache("m", "production").get()


def test_ttl_skips_the_registry_lookup_within_the_window(monkeypatch):
    registry = FakeRegistry(["1", "2"])
    registry.install(monkeypatch)
    cache = AliasedModelCache("m", "production", ttl_seconds=60)

    cache.get()
    _, version = cache.get()

    assert registry.alias_lookups == 1
    assert version == "1"


def test_ttl_expiry_triggers_a_fresh_lookup(monkeypatch):
    registry = FakeRegistry(["1", "2"])
    registry.install(monkeypatch)
    clock = {"t": 100.0}
    monkeypatch.setattr(cache_mod.time, "monotonic", lambda: clock["t"])
    cache = AliasedModelCache("m", "production", ttl_seconds=30)

    cache.get()
    clock["t"] = 131.0
    _, version = cache.get()

    assert registry.alias_lookups == 2
    assert version == "2"


def test_concurrent_first_requests_load_the_model_once(monkeypatch):
    registry = FakeRegistry(["1"])
    registry.install(monkeypatch)
    cache = AliasedModelCache("m", "production", ttl_seconds=60)
    barrier = threading.Barrier(8)
    results = []

    def worker():
        barrier.wait()
        results.append(cache.get())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(registry.loads) == 1
    assert {version for _, version in results} == {"1"}
