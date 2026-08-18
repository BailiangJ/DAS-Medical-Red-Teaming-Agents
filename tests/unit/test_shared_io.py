from dataclasses import dataclass
from pathlib import Path

import pytest

import med_red_team.shared.io as shared_io


REPO_ROOT = Path(__file__).resolve().parents[2]

from med_red_team.shared.io import (
    atomic_write_json,
    build_result_envelope,
    read_json,
    require_distinct_paths,
    validate_output_path,
)


@dataclass
class Config:
    value: int


def test_validation_does_not_touch_final_path(tmp_path):
    output = tmp_path / "result.json"
    assert validate_output_path(output) == output
    assert not output.exists()


def test_require_distinct_paths_rejects_resolved_and_hardlink_aliases(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}")
    hardlink = tmp_path / "hardlink.json"
    hardlink.hardlink_to(source)

    with pytest.raises(ValueError, match="different files"):
        require_distinct_paths(source, tmp_path / "." / "source.json")
    with pytest.raises(ValueError, match="different files"):
        require_distinct_paths(source, hardlink)
    require_distinct_paths(source, tmp_path / "other.json")


def test_existing_output_requires_overwrite(tmp_path):
    output = tmp_path / "result.json"
    output.write_text("original")
    with pytest.raises(FileExistsError):
        validate_output_path(output)
    assert output.read_text() == "original"
    assert validate_output_path(output, overwrite=True) == output


def test_hallucination_artifact_root_is_immutable_for_output_validation():
    output = REPO_ROOT / "artifacts" / "hallucination" / "runtime_output.json"
    with pytest.raises(ValueError, match="bundled release artifacts are read-only"):
        validate_output_path(output, overwrite=True)
    assert not output.exists()


def test_atomic_write_rejects_immutable_hallucination_artifact_root():
    output = REPO_ROOT / "artifacts" / "hallucination" / "runtime_output.json"
    with pytest.raises(ValueError, match="bundled release artifacts are read-only"):
        atomic_write_json(output, {"status": "blocked"})
    assert not output.exists()


def test_immutable_root_rejects_symlinked_parent_without_external_write(
    tmp_path,
    monkeypatch,
):
    immutable_root = tmp_path / "artifacts" / "hallucination"
    outside = tmp_path / "outside"
    immutable_root.mkdir(parents=True)
    outside.mkdir()
    (immutable_root / "escaped").symlink_to(outside, target_is_directory=True)
    output = immutable_root / "escaped" / "runtime_output.json"
    monkeypatch.setattr(
        shared_io,
        "_IMMUTABLE_HALLUCINATION_ROOT",
        immutable_root.resolve(),
    )

    with pytest.raises(ValueError, match="bundled release artifacts are read-only"):
        validate_output_path(output, overwrite=True)
    with pytest.raises(ValueError, match="bundled release artifacts are read-only"):
        atomic_write_json(output, {"status": "blocked"})

    assert not (outside / "runtime_output.json").exists()


def test_immutable_root_rejects_external_alias_resolving_into_bundle(
    tmp_path,
    monkeypatch,
):
    immutable_root = tmp_path / "artifacts" / "hallucination"
    immutable_root.mkdir(parents=True)
    alias = tmp_path / "bundle-alias"
    alias.symlink_to(immutable_root, target_is_directory=True)
    output = alias / "runtime_output.json"
    monkeypatch.setattr(
        shared_io,
        "_IMMUTABLE_HALLUCINATION_ROOT",
        immutable_root.resolve(),
    )

    with pytest.raises(ValueError, match="bundled release artifacts are read-only"):
        atomic_write_json(output, {"status": "blocked"})

    assert not (immutable_root / "runtime_output.json").exists()


def test_immutable_root_rejects_alias_when_final_file_symlinks_outside(
    tmp_path,
    monkeypatch,
):
    immutable_root = tmp_path / "artifacts" / "hallucination"
    outside = tmp_path / "outside"
    immutable_root.mkdir(parents=True)
    outside.mkdir()
    target = outside / "target.json"
    target.write_text('{"original": true}\n', encoding="utf-8")
    protected_link = immutable_root / "runtime_output.json"
    protected_link.symlink_to(target)
    alias = tmp_path / "bundle-alias"
    alias.symlink_to(immutable_root, target_is_directory=True)
    monkeypatch.setattr(
        shared_io,
        "_IMMUTABLE_HALLUCINATION_ROOT",
        immutable_root.resolve(),
    )

    with pytest.raises(ValueError, match="bundled release artifacts are read-only"):
        atomic_write_json(alias / protected_link.name, {"status": "blocked"})

    assert protected_link.is_symlink()
    assert target.read_text(encoding="utf-8") == '{"original": true}\n'


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
            "source": {},
            "is_partial": False,
        },
        summary={"rate": 0.5},
        data=[{"id": 1}],
        data_key="attack_results",
    )
    assert envelope["summary"]["rate"] == 0.5
    assert envelope["attack_results"] == [{"id": 1}]


def test_envelope_rejects_missing_or_unstable_common_metadata():
    base = {
        "schema_version": "2.0",
        "axis": "healthbench",
        "phase": "baseline",
        "config": {},
        "models": {},
        "dataset": {},
        "source": {},
        "is_partial": False,
    }

    for missing in base:
        metadata = dict(base)
        metadata.pop(missing)
        with pytest.raises(ValueError, match=missing):
            build_result_envelope(metadata=metadata, summary={}, data=[])

    metadata = dict(base)
    metadata["dataset"] = "data/healthbench.jsonl"
    with pytest.raises(TypeError, match="dataset"):
        build_result_envelope(metadata=metadata, summary={}, data=[])
