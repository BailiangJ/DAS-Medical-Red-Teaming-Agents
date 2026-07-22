from dataclasses import dataclass
from pathlib import Path

import pytest

from med_red_team.shared.io import (
    atomic_write_json,
    build_result_envelope,
    read_json,
    validate_output_path,
)


@dataclass
class Config:
    value: int


def test_validation_does_not_touch_final_path(tmp_path):
    output = tmp_path / "result.json"
    assert validate_output_path(output) == output
    assert not output.exists()


def test_existing_output_requires_overwrite(tmp_path):
    output = tmp_path / "result.json"
    output.write_text("original")
    with pytest.raises(FileExistsError):
        validate_output_path(output)
    assert output.read_text() == "original"
    assert validate_output_path(output, overwrite=True) == output


def test_atomic_write_serializes_dataclasses_and_paths(tmp_path):
    output = tmp_path / "result.json"
    atomic_write_json(output, {"config": Config(2), "path": Path("data/input.json")})
    assert read_json(output) == {
        "config": {"value": 2},
        "path": "data/input.json",
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_minimal_envelope_keeps_axis_payload_local():
    envelope = build_result_envelope(
        metadata={
            "schema_version": "2.0",
            "axis": "privacy",
            "phase": "attack",
            "config": {},
            "models": {},
            "dataset": {},
        },
        summary={"rate": 0.5},
        data=[{"id": 1}],
        data_key="attack_results",
    )
    assert envelope["summary"]["rate"] == 0.5
    assert envelope["attack_results"] == [{"id": 1}]
