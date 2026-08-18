from dataclasses import dataclass

import pytest

from med_red_team.shared.checkpoints import (
    load_checkpoint,
    merge_unique,
    remove_checkpoint,
    require_complete_artifact,
    retire_foreign_checkpoint,
    validate_resume_metadata,
    write_checkpoint,
)
from med_red_team.shared.config_loading import (
    clone_config,
    load_config,
    resolve_sample_limit,
)


@dataclass
class ExampleConfig:
    name: str


def test_config_loader_requires_explicit_config(tmp_path):
    good = tmp_path / "good.py"
    good.write_text(
        "from tests.unit.test_config_and_checkpoints import ExampleConfig\n"
        "CONFIG = ExampleConfig(name='ok')\n"
    )
    missing = tmp_path / "missing.py"
    missing.write_text("VALUE = 1\n")

    assert load_config(good, ExampleConfig).name == "ok"
    with pytest.raises(ValueError, match="must export CONFIG"):
        load_config(missing, ExampleConfig)


def test_clone_config_returns_independent_instance():
    original = ExampleConfig(name="original")
    cloned = clone_config(original)
    cloned.name = "changed"

    assert cloned is not original
    assert original.name == "original"


def test_sample_limit_resolution_uses_cli_then_config_and_rejects_bool():
    assert resolve_sample_limit(None, None) is None
    assert resolve_sample_limit(None, 4) == 4
    assert resolve_sample_limit(0, 4) == 0
    assert resolve_sample_limit(2, 4) == 2

    with pytest.raises(ValueError):
        resolve_sample_limit(True, None)
    with pytest.raises(ValueError):
        resolve_sample_limit(-1, None)


def test_checkpoint_primitives_and_axis_owned_identity(tmp_path):
    output = tmp_path / "results.json"
    payload = {"metadata": {"axis": "bias"}, "results": [{"id": 1}]}
    write_checkpoint(output, payload)
    assert load_checkpoint(output) == payload

    merged = merge_unique(
        [{"id": 1, "value": "old"}],
        [{"id": 1, "value": "new"}, {"id": 2, "value": "new"}],
        key=lambda item: item["id"],
    )
    assert merged == [
        {"id": 1, "value": "new"},
        {"id": 2, "value": "new"},
    ]

    remove_checkpoint(output)
    assert load_checkpoint(output) is None


def test_foreign_checkpoint_retirement_requires_explicit_opt_in(tmp_path, monkeypatch):
    foreign = tmp_path / "old.json.inprogress"
    canonical = tmp_path / "new.json.inprogress"
    foreign.write_text("old", encoding="utf-8")

    assert not retire_foreign_checkpoint(foreign, canonical)
    assert foreign.exists()

    canonical.write_text("new", encoding="utf-8")
    assert not retire_foreign_checkpoint(foreign, canonical)
    assert foreign.exists()
    assert retire_foreign_checkpoint(foreign, canonical, enabled=True)
    assert not foreign.exists()
    assert canonical.exists()

    assert not retire_foreign_checkpoint(canonical, canonical, enabled=True)
    monkeypatch.chdir(tmp_path)
    assert not retire_foreign_checkpoint(
        canonical.name,
        canonical.resolve(),
        enabled=True,
    )
    assert canonical.exists()
    complete = tmp_path / "complete.json"
    complete.write_text("complete", encoding="utf-8")
    assert not retire_foreign_checkpoint(complete, canonical, enabled=True)
    assert complete.exists()


def test_partial_artifact_is_rejected_as_completed_input():
    require_complete_artifact({"is_partial": False})
    with pytest.raises(ValueError, match="complete"):
        require_complete_artifact({"is_partial": True})
    with pytest.raises(ValueError, match="complete"):
        require_complete_artifact({})


def test_resume_metadata_validation():
    validate_resume_metadata(
        {"schema_version": "2.0", "axis": "privacy", "phase": "baseline"},
        {"schema_version": "2.0", "axis": "privacy", "phase": "baseline"},
    )
    with pytest.raises(ValueError, match="phase"):
        validate_resume_metadata(
            {"schema_version": "2.0", "axis": "privacy", "phase": "baseline"},
            {"schema_version": "2.0", "axis": "privacy", "phase": "attack"},
        )
