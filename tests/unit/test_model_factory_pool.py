from dataclasses import replace

import pytest

from med_red_team.model_pool import ModelPool
from med_red_team.models.factory import ModelFactory
from med_red_team.models.interfaces import InfrastructureConfig, LLMInterface, ModelResponse


class DummyModel(LLMInterface):
    def __init__(self, model_id, infra_config=None, **kwargs):
        super().__init__(model_id)
        self.infra_config = infra_config
        self.kwargs = kwargs
        self.closed = False

    def generate(self, user_prompt, system_prompt, config):
        return ModelResponse(raw_text="ok", model_id=self.model_id)

    def close(self):
        self.closed = True


def test_factory_overrides_do_not_mutate_registry():
    default = InfrastructureConfig(tensor_parallel_size=2)
    previous = ModelFactory._registry
    ModelFactory.register_models({
        "dummy": {"class": DummyModel, "kwargs": {"infra_config": default}},
    })
    try:
        first = ModelFactory.create("dummy", tensor_parallel_size=4)
        second = ModelFactory.create("dummy")
    finally:
        ModelFactory.register_models(previous)

    assert first.infra_config.tensor_parallel_size == 4
    assert second.infra_config.tensor_parallel_size == 2
    assert default.tensor_parallel_size == 2


def test_pool_identity_uses_construction_overrides(monkeypatch):
    created = []

    def create(model_id, **kwargs):
        model = DummyModel(model_id, **kwargs)
        created.append(model)
        return model

    monkeypatch.setattr(ModelFactory, "create", create)
    pool = ModelPool()

    first = pool.get_model("dummy", seed=1)
    same = pool.get_model("dummy", seed=1)
    different = pool.get_model("dummy", seed=2)

    assert first is same
    assert different is not first
    assert len(created) == 2
    assert pool.is_loaded("dummy", seed=1)


def test_pool_rejects_generation_settings():
    with pytest.raises(ValueError, match="Generation settings"):
        ModelPool().get_model("dummy", temperature=0.2)


def test_pool_clear_calls_close(monkeypatch):
    model = DummyModel("dummy")
    monkeypatch.setattr(ModelFactory, "create", lambda model_id, **kwargs: model)
    pool = ModelPool()
    pool.get_model("dummy")

    pool.clear()

    assert model.closed
    assert len(pool) == 0
