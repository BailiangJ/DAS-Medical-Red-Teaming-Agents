from dataclasses import dataclass

import pytest

from med_red_team.shared.checkpoints import (
    load_checkpoint,
    merge_unique,
    remove_checkpoint,
    validate_resume_metadata,
    write_checkpoint,
)
from med_red_team.shared.config_loading import load_config


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
