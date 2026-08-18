from __future__ import annotations

import json
from pathlib import Path

import pytest

from med_red_team.hallucination.config import (
    ArtifactSummaryConfig,
    DetectorExecutionConfig,
    StanfordResponseGenerationConfig,
    load_config,
    resolve_max_samples,
    validate_max_samples,
)
from med_red_team.hallucination.data import (
    HallucinationDataError,
    load_detector_outputs,
    load_generated_response_detector_list,
    load_legacy_generated_response_envelope,
    load_native_detector_output_list,
    load_native_prompt_response_json,
    load_prompt_responses,
)
from med_red_team.models import GenerationConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "hallucination"
DATASET_ROOT = ARTIFACT_ROOT / "datasets"


@pytest.mark.parametrize(
    ("max_samples", "total", "expected"),
    [
        (None, 7, 7),
        (0, 7, 0),
        (3, 7, 3),
        (99, 7, 7),
        (None, 0, 0),
    ],
)
def test_max_samples_semantics(max_samples, total, expected):
    assert resolve_max_samples(max_samples, total) == expected


@pytest.mark.parametrize("bad_value", [-1, -42, True, False, 1.5])
def test_max_samples_rejects_negative_values(bad_value):
    with pytest.raises(ValueError, match="max_samples must be None, 0, or a positive integer"):
        validate_max_samples(bad_value)


def test_config_objects_round_trip_through_dict_and_json(tmp_path):
    configs = [
        StanfordResponseGenerationConfig(
            dataset_path="input.json",
            output_dir="out",
            model_ids=["fake-model"],
            generation_config=GenerationConfig(temperature=0.2, max_tokens=17),
            max_samples=0,
            metadata={"purpose": "roundtrip"},
        ),
        DetectorExecutionConfig(
            input_path="input.json",
            output_path="output.json",
            backend="claude",
            input_format="generated",
            max_samples=5,
            start_idx=2,
            end_idx=9,
            metadata={"purpose": "roundtrip"},
        ),
        ArtifactSummaryConfig(
            artifact_root="artifacts/hallucination",
            manifest_name="manifest.json",
            max_samples=None,
            raise_on_error=True,
        ),
    ]
    classes = [StanfordResponseGenerationConfig, DetectorExecutionConfig, ArtifactSummaryConfig]

    for index, (config, cls) in enumerate(zip(configs, classes)):
        as_dict = config.to_dict()
        assert cls.from_dict(as_dict).to_dict() == as_dict

        json_path = tmp_path / f"config_{index}.json"
        json_path.write_text(json.dumps(as_dict), encoding="utf-8")
        assert load_config(json_path, cls).to_dict() == as_dict


@pytest.mark.parametrize(
    ("path", "cls"),
    [
        (REPO_ROOT / "configs" / "examples" / "hallucination" / "response_generation.py", StanfordResponseGenerationConfig),
        (REPO_ROOT / "configs" / "examples" / "hallucination" / "detector_validation_stanford_positive_131.py", DetectorExecutionConfig),
        (REPO_ROOT / "configs" / "examples" / "hallucination" / "detector_validation_healthbench_negative_131.py", DetectorExecutionConfig),
        (REPO_ROOT / "configs" / "examples" / "hallucination" / "artifact_summary.py", ArtifactSummaryConfig),
        (REPO_ROOT / "configs" / "paper" / "hallucination" / "response_generation.py", StanfordResponseGenerationConfig),
        (REPO_ROOT / "configs" / "paper" / "hallucination" / "detector_validation_stanford_positive_131.py", DetectorExecutionConfig),
        (REPO_ROOT / "configs" / "paper" / "hallucination" / "detector_validation_healthbench_negative_131.py", DetectorExecutionConfig),
        (REPO_ROOT / "configs" / "paper" / "hallucination" / "artifact_summary.py", ArtifactSummaryConfig),
    ],
)
def test_checked_in_config_exports_round_trip(path, cls):
    config = load_config(path, cls)
    assert isinstance(config, cls)
    assert cls.from_dict(config.to_dict()).to_dict() == config.to_dict()


def test_split_detector_presets_are_explicit_and_unambiguous():
    stanford_paper = load_config(
        REPO_ROOT / "configs" / "paper" / "hallucination" / "detector_validation_stanford_positive_131.py",
        DetectorExecutionConfig,
    )
    healthbench_paper = load_config(
        REPO_ROOT / "configs" / "paper" / "hallucination" / "detector_validation_healthbench_negative_131.py",
        DetectorExecutionConfig,
    )
    stanford_example = load_config(
        REPO_ROOT / "configs" / "examples" / "hallucination" / "detector_validation_stanford_positive_131.py",
        DetectorExecutionConfig,
    )
    healthbench_example = load_config(
        REPO_ROOT / "configs" / "examples" / "hallucination" / "detector_validation_healthbench_negative_131.py",
        DetectorExecutionConfig,
    )

    assert stanford_paper.input_path.endswith("stanford_redteaming_131_positive_cases.json")
    assert healthbench_paper.input_path.endswith("healthbench_131_negative_cases.json")
    assert stanford_paper.max_samples is None
    assert healthbench_paper.max_samples is None
    assert (stanford_paper.backend, stanford_paper.orchestrator_model, stanford_paper.sub_agent_model) == (
        "openai",
        "gpt-4o",
        "o3",
    )
    assert (healthbench_paper.backend, healthbench_paper.orchestrator_model, healthbench_paper.sub_agent_model) == (
        "openai",
        "gpt-4o",
        "o3",
    )
    assert stanford_example.max_samples == 2
    assert healthbench_example.max_samples == 2


def test_detector_row_slice_uses_start_end_and_max_samples_semantics():
    all_rows = DetectorExecutionConfig(input_path="in.json", output_path="out.json", max_samples=None)
    assert all_rows.resolve_row_slice(10) == slice(0, 10)

    no_work = DetectorExecutionConfig(input_path="in.json", output_path="out.json", max_samples=0)
    assert no_work.resolve_row_slice(10) == slice(0, 0)

    capped = DetectorExecutionConfig(
        input_path="in.json",
        output_path="out.json",
        start_idx=2,
        end_idx=8,
        max_samples=3,
    )
    assert capped.resolve_row_slice(10) == slice(2, 5)


def test_detector_config_uses_backend_specific_model_defaults():
    openai = DetectorExecutionConfig(input_path="in.json", output_path="out.json")
    claude = DetectorExecutionConfig(
        input_path="in.json",
        output_path="out.json",
        backend="claude",
    )

    assert (openai.orchestrator_model, openai.sub_agent_model) == ("gpt-4o", "gpt-4o")
    assert (claude.orchestrator_model, claude.sub_agent_model) == (
        "claude-sonnet-4-20250514",
        "inherit",
    )


@pytest.mark.parametrize(
    ("path", "expected_count", "first_identity"),
    [
        (DATASET_ROOT / "stanford_redteaming_131_positive_cases.json", 131, "0"),
        (DATASET_ROOT / "healthbench_131_negative_cases.json", 131, "0"),
    ],
)
def test_native_prompt_response_loaders_cover_packaged_datasets(path, expected_count, first_identity):
    records = load_native_prompt_response_json(path)

    assert len(records) == expected_count
    assert records[0].identity == first_identity
    assert records[0].identity == str(records[0].row_idx)
    assert records[0].source_format == "native"
    assert len({record.identity for record in records}) == expected_count
    assert all(record.prompt and record.response for record in records)

    assert load_native_prompt_response_json(path, max_samples=0) == []
    assert len(load_native_prompt_response_json(path, max_samples=2)) == 2
    assert len(load_native_prompt_response_json(path, max_samples=None)) == expected_count


def test_legacy_generated_response_envelope_loader_has_model_ordinal_identities():
    path = ARTIFACT_ROOT / "generated_responses" / "stanford_positive_131" / "results_gpt-4o_20250626_122402.json"

    envelope = load_legacy_generated_response_envelope(path)

    assert envelope.path == str(path)
    assert len(envelope.responses) == 131
    assert envelope.statistics["total_prompts"] == 131
    assert envelope.responses[0].model == "gpt-4o"
    assert envelope.responses[0].identity == "gpt-4o:0"
    assert envelope.responses[-1].identity == "gpt-4o:130"
    assert len({record.identity for record in envelope.responses}) == 131

    limited = load_prompt_responses(path, source_format="auto", max_samples=3)
    assert limited.responses[0].source_format == "generated"
    assert [record.identity for record in limited.responses] == ["gpt-4o:0", "gpt-4o:1", "gpt-4o:2"]


def test_native_and_generated_detector_loaders_infer_formats_and_stable_identities():
    native_path = ARTIFACT_ROOT / "native_detector" / "openai_gpt4o_o3" / "stanford_positive_131" / "detailed_outputs.json"
    generated_path = ARTIFACT_ROOT / "generated_response_detection" / "openai_gpt4o_o3" / "gpt_4o_positive" / "detailed_outputs.json"

    native = load_detector_outputs(native_path, source_format="auto")
    generated = load_detector_outputs(generated_path, source_format="auto")

    assert native.source_format == "native"
    assert len(native.records) == 131
    assert native.records[0].identity == str(native.records[0].row_idx)
    assert native.records[0].source_format == "native"
    assert len({record.identity for record in native.records}) == 131

    assert generated.source_format == "generated"
    assert len(generated.records) == 130
    assert generated.records[0].identity == f"{generated.records[0].model}:artifact_row:0"
    assert generated.records[0].ordinal is None
    assert generated.records[-1].identity == f"{generated.records[-1].model}:artifact_row:129"
    assert generated.records[0].source_format == "generated"
    assert len({record.identity for record in generated.records}) == 130

    assert load_native_detector_output_list(native_path, max_samples=0) == []
    assert len(load_generated_response_detector_list(generated_path, max_samples=2)) == 2


def test_data_loaders_raise_helpful_invalid_shape_errors(tmp_path):
    not_a_list = tmp_path / "not_a_list.json"
    not_a_list.write_text(json.dumps({"Prompt": "p"}), encoding="utf-8")
    with pytest.raises(HallucinationDataError, match="Expected top-level JSON.*to be a list"):
        load_native_prompt_response_json(not_a_list)

    missing_row_idx = tmp_path / "missing_row_idx.json"
    missing_row_idx.write_text(json.dumps([{"Prompt": "p", "Response": "r"}]), encoding="utf-8")
    with pytest.raises(HallucinationDataError, match="Missing required key 'row_idx'.*row 0"):
        load_native_prompt_response_json(missing_row_idx)

    bad_generated = tmp_path / "bad_generated.json"
    bad_generated.write_text(json.dumps({"results": []}), encoding="utf-8")
    with pytest.raises(HallucinationDataError, match="must contain 'results' and 'statistics'"):
        load_legacy_generated_response_envelope(bad_generated, model="fake")

    bad_detector = tmp_path / "bad_detector.json"
    bad_detector.write_text(
        json.dumps([
            {
                "prompt": "p",
                "response": "r",
                "merged_codes": "0",
                "rationale": None,
                "agent_decisions": {},
                "metadata": {"model": "fake"},
            }
        ]),
        encoding="utf-8",
    )
    with pytest.raises(HallucinationDataError, match="agent_decisions must be a list.*row 0"):
        load_detector_outputs(bad_detector, source_format="generated")


def test_data_loaders_reject_duplicate_stable_identities(tmp_path):
    duplicate_native = tmp_path / "duplicate_native.json"
    duplicate_native.write_text(
        json.dumps([
            {"Prompt": "p1", "Response": "r1", "row_idx": 3},
            {"Prompt": "p2", "Response": "r2", "row_idx": 3},
        ]),
        encoding="utf-8",
    )

    with pytest.raises(HallucinationDataError, match="Duplicate stable identities.*3"):
        load_native_prompt_response_json(duplicate_native)
