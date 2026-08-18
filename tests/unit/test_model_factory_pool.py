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


def test_pool_reuses_default_equivalent_effective_config():
    default = InfrastructureConfig(tensor_parallel_size=2, max_model_len=8192)
    previous = ModelFactory._registry
    ModelFactory.register_models({
        "dummy": {"class": DummyModel, "kwargs": {"infra_config": default}},
    })
    try:
        pool = ModelPool()
        implicit = pool.get_model("dummy")
        explicit = pool.get_model(
            "dummy",
            tensor_parallel_size=2,
            max_model_len=8192,
        )
    finally:
        pool.clear()
        ModelFactory.register_models(previous)

    assert implicit is explicit


def test_factory_rejects_mixed_full_and_scalar_infrastructure_overrides():
    previous = ModelFactory._registry
    ModelFactory.register_models({
        "dummy": {
            "class": DummyModel,
            "kwargs": {"infra_config": InfrastructureConfig()},
        },
    })
    try:
        with pytest.raises(ValueError, match="infra_config"):
            ModelFactory.create(
                "dummy",
                infra_config=InfrastructureConfig(tensor_parallel_size=4),
                tensor_parallel_size=2,
            )
    finally:
        ModelFactory.register_models(previous)


def test_pool_identity_uses_construction_overrides():
    previous = ModelFactory._registry
    ModelFactory.register_models({
        "dummy": {"class": DummyModel, "kwargs": {}},
    })
    try:
        pool = ModelPool()
        first = pool.get_model("dummy", endpoint="one")
        same = pool.get_model("dummy", endpoint="one")
        different = pool.get_model("dummy", endpoint="two")

        assert first is same
        assert different is not first
        assert len(pool) == 2
        assert pool.is_loaded("dummy", endpoint="one")
    finally:
        pool.clear()
        ModelFactory.register_models(previous)


def test_pool_rejects_generation_settings():
    with pytest.raises(ValueError, match="Generation settings"):
        ModelPool().get_model("dummy", temperature=0.2)


def test_pool_clear_calls_close():
    previous = ModelFactory._registry
    ModelFactory.register_models({
        "dummy": {"class": DummyModel, "kwargs": {}},
    })
    try:
        pool = ModelPool()
        model = pool.get_model("dummy")
        pool.clear()
    finally:
        ModelFactory.register_models(previous)

    assert model.closed
    assert len(pool) == 0


def test_pool_cache_events_do_not_print(capsys):
    previous = ModelFactory._registry
    ModelFactory.register_models({
        "dummy": {"class": DummyModel, "kwargs": {}},
    })
    try:
        pool = ModelPool()
        pool.get_model("dummy")
        pool.get_model("dummy")
        pool.clear()
    finally:
        ModelFactory.register_models(previous)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
