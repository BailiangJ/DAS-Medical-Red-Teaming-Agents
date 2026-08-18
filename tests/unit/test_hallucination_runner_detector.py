from __future__ import annotations

import builtins
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from med_red_team.hallucination import pipeline as hallucination_pipeline
from med_red_team.hallucination.config import DetectorExecutionConfig, StanfordResponseGenerationConfig
from med_red_team.hallucination.detectors import row_identity
from med_red_team.hallucination.detectors import claude_agent_sdk, openai_agents
from med_red_team.hallucination.pipeline import (
    HallucinationDetectionPipeline,
    HallucinationResponseGenerationPipeline,
)
from med_red_team.models import GenerationConfig, ModelExecutionError, ModelResponse
from med_red_team.shared.checkpoints import checkpoint_path
from med_red_team.shared.io import read_json
from scripts.hallucination import run_baseline, run_detector


REPO_ROOT = Path(__file__).resolve().parents[2]
STANFORD_ARTIFACT = REPO_ROOT / "artifacts" / "hallucination" / "datasets" / "stanford_redteaming_131_positive_cases.json"


def test_openai_detector_accepts_custom_model_ids_without_allowlist():
    detector = openai_agents.OpenAIAgentsHallucinationDetector(
        orchestrator_model="gpt-5-future",
        sub_agent_model="gpt-5-future-mini",
        search_agent_model="gpt-5-future-search",
    )

    assert detector.to_dict()["orchestrator_model"] == "gpt-5-future"
    assert detector.to_dict()["sub_agent_model"] == "gpt-5-future-mini"
    assert detector.to_dict()["search_agent_model"] == "gpt-5-future-search"


def test_generation_cli_rejects_unknown_options():
    with pytest.raises(SystemExit):
        run_baseline.parse_args(["--paper-mode"])


def test_detector_warns_when_external_retrieval_is_unavailable():
    with pytest.warns(RuntimeWarning, match="External retrieval is unavailable") as caught:
        openai_agents.OpenAIAgentsHallucinationDetector(
            search_agent_model="o3",
            orchestrator=object(),
        )

    assert len(caught) == 1


def _native_rows(count: int) -> list[dict[str, Any]]:
    return [
        {
            "Prompt": f"prompt {index}",
            "Response": f"source response {index}",
            "row_idx": index,
            "Hallucination/Accuracy": 1.0,
            "Model": "source-model",
            "Additional comments": None,
        }
        for index in range(count)
    ]


def _write_dataset(tmp_path: Path, count: int, *, name: str = "dataset.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(_native_rows(count)), encoding="utf-8")
    return path


def _generation_config(
    dataset_path: Path,
    output_dir: Path,
    *,
    max_samples: int | None = None,
    overwrite: bool = True,
) -> StanfordResponseGenerationConfig:
    return StanfordResponseGenerationConfig(
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
        model_ids=["fake-model"],
        system_prompt="fake system",
        generation_config=GenerationConfig(temperature=0.0, max_tokens=32),
        max_samples=max_samples,
        overwrite=overwrite,
    )


class _CountingModel:
    def __init__(
        self,
        responses: list[ModelResponse] | None = None,
        *,
        fail_on: set[str] | None = None,
        failure_map: dict[str, Exception] | None = None,
    ):
        self.model_id = "fake-model"
        self.responses = list(responses or [])
        self.fail_on = fail_on or set()
        self.failure_map = dict(failure_map or {})
        self.calls: list[str] = []

    def generate(self, user_prompt: str, system_prompt: str, config: GenerationConfig) -> ModelResponse:
        self.calls.append(user_prompt)
        if user_prompt in self.failure_map:
            raise self.failure_map[user_prompt]
        if user_prompt in self.fail_on:
            raise RuntimeError(f"boom for {user_prompt}")
        if self.responses:
            return self.responses.pop(0)
        return ModelResponse(
            raw_text=f"raw {user_prompt}",
            final_answer=f"answer {user_prompt}",
            reasoning="",
            model_id=self.model_id,
            metadata={"provider": "fake"},
            generation_config=config,
        )


class _CountingPool:
    created = 0
    cleared = 0
    get_model_calls: list[str] = []
    model: _CountingModel

    def __init__(self) -> None:
        type(self).created += 1

    def get_model(self, model_id: str):
        type(self).get_model_calls.append(model_id)
        return type(self).model

    def clear(self) -> None:
        type(self).cleared += 1

    @classmethod
    def reset(cls, model: _CountingModel) -> None:
        cls.created = 0
        cls.cleared = 0
        cls.get_model_calls = []
        cls.model = model


def _build_generation_plan(
    config: StanfordResponseGenerationConfig,
) -> dict[str, Any]:
    return HallucinationResponseGenerationPipeline().build_generation_plan(config)


def _generation_pipeline(*, model_pool_factory=_CountingPool) -> HallucinationResponseGenerationPipeline:
    return HallucinationResponseGenerationPipeline(model_pool_factory=model_pool_factory)


def _generation_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prompt": row["Prompt"],
            "response": f"existing answer {row['row_idx']}",
            "raw_response": f"existing raw {row['row_idx']}",
            "reasoning": None,
            "row_idx": row["row_idx"],
            "sample_id": str(row["row_idx"]),
            "metadata": {"row_idx": row["row_idx"], "model": "fake-model"},
            "error": None,
        }
        for row in rows
    ]


def _generation_envelope(
    config: StanfordResponseGenerationConfig,
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    is_partial: bool,
) -> dict[str, Any]:
    return {
        "metadata": {
            "schema_version": "2.0",
            "axis": "hallucination",
            "phase": "response_generation",
            "models": {"target_model": "fake-model"},
            "dataset": {
                "sha256": plan["dataset_sha256"],
                "planned_count": plan["planned_count"],
                "planned_prompts_sha256": plan["planned_prompts_sha256"],
            },
            "source": {"fixture": "generation"},
            "config": config.to_dict(),
            "is_partial": is_partial,
        },
        "summary": {
            "planned_count": len(rows),
            "result_count": len(rows),
            "completed_count": len(rows),
            "failed_count": 0,
        },
        "results": _generation_results(rows),
    }


def test_validate_only_does_not_construct_model_or_write(tmp_path, monkeypatch, capsys):
    dataset = _write_dataset(tmp_path, 2)
    config = _generation_config(dataset, tmp_path / "outputs", max_samples=None)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")

    class ExplodingModelPool:
        def __init__(self):
            raise AssertionError("validate-only must not construct ModelPool")

    def fail_write(*args, **kwargs):
        raise AssertionError("validate-only must not write output artifacts")

    monkeypatch.setattr(hallucination_pipeline, "ModelPool", ExplodingModelPool)
    monkeypatch.setattr(hallucination_pipeline, "atomic_write_json", fail_write)

    assert run_baseline.main(["--config", str(config_path), "--validate-only", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["validation"]["planned_count"] == 2
    assert payload["validation"]["total_rows"] == 2
    assert not (tmp_path / "outputs").exists()


def test_generation_constructs_one_model_for_multiple_prompts(tmp_path, monkeypatch):
    dataset = _write_dataset(tmp_path, 3)
    config = _generation_config(dataset, tmp_path / "outputs")
    plan = _build_generation_plan(config)
    output_path = tmp_path / "outputs" / "results.json"
    model = _CountingModel()
    _CountingPool.reset(model)

    result = _generation_pipeline().run(
        config,
        output_path=output_path,
        plan=plan,
        quiet=True,
        save_every=0,
    )

    assert _CountingPool.created == 1
    assert _CountingPool.get_model_calls == ["fake-model"]
    assert _CountingPool.cleared == 1
    assert model.calls == ["prompt 0", "prompt 1", "prompt 2"]
    assert result["completed_count"] == 3
    assert read_json(output_path)["summary"]["total_responses"] == 3


def test_max_samples_zero_skips_model_construction_and_output(tmp_path, monkeypatch):
    dataset = _write_dataset(tmp_path, 2)
    config = _generation_config(dataset, tmp_path / "outputs", max_samples=0)
    plan = _build_generation_plan(config)
    output_path = tmp_path / "outputs" / "results.json"

    class ExplodingModelPool:
        def __init__(self):
            raise AssertionError("max_samples=0 must not construct ModelPool")

    result = _generation_pipeline(model_pool_factory=ExplodingModelPool).run(
        config,
        output_path=output_path,
        plan=plan,
        quiet=True,
        save_every=0,
    )

    assert result["planned_count"] == 0
    assert result["output_path"] is None
    assert "no model was constructed" in result["message"]
    assert not output_path.exists()


def test_max_samples_zero_json_output_is_machine_readable(tmp_path, capsys):
    dataset = _write_dataset(tmp_path, 1)
    config = _generation_config(dataset, tmp_path / "outputs", max_samples=0)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")

    assert run_baseline.main(["--config", str(config_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["planned_count"] == 0


def test_generation_complete_existing_output_short_circuits_before_model_construction(tmp_path, monkeypatch):
    dataset = _write_dataset(tmp_path, 2)
    config = _generation_config(dataset, tmp_path / "outputs", overwrite=False)
    plan = _build_generation_plan(config)
    output_path = tmp_path / "outputs" / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_generation_envelope(config, plan, _native_rows(2), is_partial=False)),
        encoding="utf-8",
    )
    original = output_path.read_bytes()

    class ExplodingModelPool:
        def __init__(self):
            raise AssertionError("complete resume must not construct ModelPool")

    result = _generation_pipeline(model_pool_factory=ExplodingModelPool).run(
        config,
        output_path=output_path,
        plan=plan,
        quiet=True,
        save_every=0,
    )

    assert result["completed_count"] == 2
    assert "already complete" in result["message"]
    assert output_path.read_bytes() == original


def test_generation_all_complete_checkpoint_finalizes_without_model_construction(tmp_path, monkeypatch):
    dataset = _write_dataset(tmp_path, 2)
    config = _generation_config(dataset, tmp_path / "outputs", overwrite=False)
    plan = _build_generation_plan(config)
    output_path = tmp_path / "outputs" / "results.json"
    partial_path = checkpoint_path(output_path)
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_text(
        json.dumps(_generation_envelope(config, plan, _native_rows(2), is_partial=True)),
        encoding="utf-8",
    )

    class ExplodingModelPool:
        def __init__(self):
            raise AssertionError("all-complete checkpoint must not construct ModelPool")

    result = _generation_pipeline(model_pool_factory=ExplodingModelPool).run(
        config,
        output_path=output_path,
        plan=plan,
        quiet=True,
        save_every=0,
    )

    payload = read_json(output_path)
    assert result["completed_count"] == 2
    assert payload["metadata"]["is_partial"] is False
    assert not partial_path.exists()
    assert "finalized without model construction" in result["message"]


def test_generation_systemic_model_errors_checkpoint_and_abort(tmp_path, monkeypatch):
    dataset = _write_dataset(tmp_path, 3)
    config = _generation_config(dataset, tmp_path / "outputs")
    plan = _build_generation_plan(config)
    output_path = tmp_path / "outputs" / "results.json"
    model = _CountingModel(failure_map={"prompt 1": ModelExecutionError("provider down")})
    _CountingPool.reset(model)

    with pytest.raises(ModelExecutionError, match="provider down"):
        _generation_pipeline().run(
            config,
            output_path=output_path,
            plan=plan,
            quiet=True,
            save_every=0,
        )

    partial = read_json(checkpoint_path(output_path))
    assert [row["row_idx"] for row in partial["results"]] == [0]
    assert partial["metadata"]["is_partial"] is True
    assert not output_path.exists()


def test_generation_rejects_immutable_artifact_output_before_model_construction(tmp_path, monkeypatch):
    dataset = _write_dataset(tmp_path, 1)
    config = _generation_config(dataset, tmp_path / "outputs")
    plan = _build_generation_plan(config)
    output_path = REPO_ROOT / "artifacts" / "hallucination" / "runtime_output.json"

    class ExplodingModelPool:
        def __init__(self):
            raise AssertionError("immutable artifact roots must fail before model construction")

    with pytest.raises(ValueError, match="bundled release artifacts are read-only"):
        _generation_pipeline(model_pool_factory=ExplodingModelPool).run(
            config,
            output_path=output_path,
            plan=plan,
            quiet=True,
            save_every=0,
        )
    assert not output_path.exists()


def test_failure_rows_are_structured_and_summary_has_no_response_time_keyerror(tmp_path, monkeypatch):
    dataset = _write_dataset(tmp_path, 2)
    config = _generation_config(dataset, tmp_path / "outputs")
    plan = _build_generation_plan(config)
    output_path = tmp_path / "outputs" / "results.json"
    model = _CountingModel(fail_on={"prompt 1"})
    _CountingPool.reset(model)

    result = _generation_pipeline().run(
        config,
        output_path=output_path,
        plan=plan,
        quiet=True,
        save_every=0,
    )

    payload = read_json(output_path)
    failed = [row for row in payload["results"] if row["error"]]
    assert result["failed_count"] == 1
    assert payload["summary"]["failed_count"] == 1
    assert failed[0]["response"] == ""
    assert failed[0]["error"] == {"type": "RuntimeError", "message": "boom for prompt 1"}
    assert isinstance(failed[0]["metadata"]["response_time"], float)


def test_generation_preserves_raw_response_when_no_final_answer_and_reasoning(tmp_path, monkeypatch):
    dataset = _write_dataset(tmp_path, 1)
    config = _generation_config(dataset, tmp_path / "outputs")
    plan = _build_generation_plan(config)
    output_path = tmp_path / "outputs" / "results.json"
    response = ModelResponse(
        raw_text="raw response body with thinking tags already stripped by fake",
        final_answer="",
        reasoning="hidden reasoning summary",
        model_id="fake-model",
        metadata={"finish_reason": "stop"},
        generation_config=config.generation_config,
    )
    model = _CountingModel(responses=[response])
    _CountingPool.reset(model)

    _generation_pipeline().run(
        config,
        output_path=output_path,
        plan=plan,
        quiet=True,
        save_every=0,
    )

    row = read_json(output_path)["results"][0]
    assert row["response"] == ""
    assert row["raw_response"] == response.raw_text
    assert row["reasoning"] == response.reasoning
    assert row["error"]["type"] == "EmptyResponseError"
    assert row["metadata"]["reasoning"] == response.reasoning
    assert row["metadata"]["model_response_metadata"] == {"finish_reason": "stop"}


def test_resume_envelope_reuses_existing_rows_by_stable_identity(tmp_path, monkeypatch):
    dataset = _write_dataset(tmp_path, 3)
    config = _generation_config(dataset, tmp_path / "outputs")
    plan = _build_generation_plan(config)
    output_path = tmp_path / "outputs" / "results.json"
    resume_path = tmp_path / "resume.json"
    resume_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "schema_version": "2.0",
                    "axis": "hallucination",
                    "phase": "response_generation",
                    "models": {"target_model": "fake-model"},
                    "dataset": {
                        "sha256": plan["dataset_sha256"],
                        "planned_prompts_sha256": plan["planned_prompts_sha256"],
                    },
                    "source": {"fixture": "resume"},
                    "config": config.to_dict(),
                    "is_partial": True,
                },
                "summary": {},
                "results": [
                    {
                        "prompt": "prompt 0",
                        "response": "existing answer",
                        "row_idx": 0,
                        "sample_id": "0",
                        "metadata": {"row_idx": 0},
                        "error": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    model = _CountingModel()
    _CountingPool.reset(model)

    _generation_pipeline().run(
        config,
        output_path=output_path,
        plan=plan,
        resume_path=resume_path,
        quiet=True,
        save_every=0,
    )

    payload = read_json(output_path)
    assert [row["row_idx"] for row in payload["results"]] == [0, 1, 2]
    assert payload["results"][0]["response"] == "existing answer"
    assert model.calls == ["prompt 1", "prompt 2"]


def test_generation_resume_accepts_reordering_and_ignores_extra_rows(tmp_path):
    dataset = _write_dataset(tmp_path, 3)
    source_rows = json.loads(dataset.read_text())
    config = _generation_config(dataset, tmp_path / "outputs")
    plan = _build_generation_plan(config)
    resume_payload = _generation_envelope(
        config,
        plan,
        [source_rows[2], source_rows[0]],
        is_partial=True,
    )
    resume_payload["results"] = [
        resume_payload["results"][0],
        {
            "prompt": "extra prompt",
            "response": "extra response",
            "row_idx": 99,
            "sample_id": "99",
            "metadata": {"row_idx": 99, "model": "fake-model"},
            "error": None,
        },
        resume_payload["results"][1],
    ]
    resume_path = tmp_path / "resume.json"
    resume_path.write_text(json.dumps(resume_payload), encoding="utf-8")
    output_path = tmp_path / "outputs" / "results.json"
    model = _CountingModel()
    _CountingPool.reset(model)

    with pytest.warns(RuntimeWarning, match="Ignoring 1 resume rows"):
        _generation_pipeline().run(
            config,
            output_path=output_path,
            plan=plan,
            resume_path=resume_path,
            quiet=True,
            save_every=0,
        )

    payload = read_json(output_path)
    assert [row["row_idx"] for row in payload["results"]] == [0, 1, 2]
    assert model.calls == ["prompt 1"]
    assert payload["summary"]["ignored_resume_count"] == 1


def test_generation_resume_retries_failed_rows(tmp_path):
    dataset = _write_dataset(tmp_path, 1)
    source_rows = json.loads(dataset.read_text())
    config = _generation_config(dataset, tmp_path / "outputs")
    plan = _build_generation_plan(config)
    resume_payload = _generation_envelope(
        config,
        plan,
        source_rows,
        is_partial=False,
    )
    resume_payload["results"][0]["response"] = ""
    resume_payload["results"][0]["error"] = {
        "type": "TemporaryError",
        "message": "retry",
    }
    resume_path = tmp_path / "resume.json"
    resume_path.write_text(json.dumps(resume_payload), encoding="utf-8")
    output_path = tmp_path / "outputs" / "results.json"
    model = _CountingModel()
    _CountingPool.reset(model)

    _generation_pipeline().run(
        config,
        output_path=output_path,
        plan=plan,
        resume_path=resume_path,
        quiet=True,
        save_every=0,
    )

    payload = read_json(output_path)
    assert model.calls == ["prompt 0"]
    assert payload["results"][0]["error"] is None
    assert payload["summary"]["successful_count"] == 1


def test_resume_requires_v2_result_envelope(tmp_path):
    dataset = _write_dataset(tmp_path, 1)
    config = _generation_config(dataset, tmp_path / "outputs")
    plan = _build_generation_plan(config)
    bad_resume = tmp_path / "bad_resume.json"
    bad_resume.write_text(json.dumps([{"row_idx": 0}]), encoding="utf-8")

    with pytest.raises(ValueError, match="Resume/checkpoint file must be a v2 envelope"):
        _generation_pipeline(model_pool_factory=lambda: (_ for _ in ()).throw(AssertionError("resume validation should fail before model construction"))).run(
            config,
            output_path=tmp_path / "outputs" / "results.json",
            plan=plan,
            resume_path=bad_resume,
            quiet=True,
            save_every=0,
        )


def test_resume_rejects_incompatible_model_metadata(tmp_path):
    dataset = _write_dataset(tmp_path, 1)
    config = _generation_config(dataset, tmp_path / "outputs")
    plan = _build_generation_plan(config)
    resume_path = tmp_path / "resume.json"
    resume_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "schema_version": "2.0",
                    "axis": "hallucination",
                    "phase": "response_generation",
                    "models": {"target_model": "different-model"},
                    "dataset": {
                        "sha256": plan["dataset_sha256"],
                        "planned_prompts_sha256": plan["planned_prompts_sha256"],
                    },
                    "source": {"fixture": "resume"},
                    "config": config.to_dict(),
                    "is_partial": True,
                },
                "summary": {},
                "results": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target model differs"):
        _generation_pipeline(model_pool_factory=lambda: (_ for _ in ()).throw(AssertionError("resume validation should fail before model construction"))).run(
            config,
            output_path=tmp_path / "outputs" / "results.json",
            plan=plan,
            resume_path=resume_path,
            quiet=True,
            save_every=0,
        )


def test_generation_plan_accepts_arbitrary_configured_dataset(tmp_path):
    dataset = tmp_path / "custom.json"
    dataset.write_text(
        json.dumps([
            {
                "question": "question",
                "reference": "reference",
                "id": "row-a",
                "Hallucination/Accuracy": "positive",
            }
        ]),
        encoding="utf-8",
    )
    config = StanfordResponseGenerationConfig(
        dataset_path=str(dataset),
        output_dir=str(tmp_path / "outputs"),
        model_ids=["fake-model"],
        prompt_key="question",
        response_key="reference",
        row_id_key="id",
    )

    plan = _build_generation_plan(config)

    assert plan["total_rows"] == 1
    assert plan["planned_identities"] == ["row-a"]
    assert plan["positive_rows"] == 0


def test_openai_detector_zero_work_does_not_import_sdk_or_write(tmp_path, monkeypatch):
    input_path = _write_dataset(tmp_path, 1)
    output_path = tmp_path / "detector.json"
    config = DetectorExecutionConfig(
        input_path=str(input_path),
        output_path=str(output_path),
        backend="openai",
        max_samples=0,
    )

    monkeypatch.setattr(
        openai_agents,
        "_import_openai_agents",
        lambda: (_ for _ in ()).throw(AssertionError("zero-work run imported SDK")),
    )
    result = HallucinationDetectionPipeline().run_native_validation(config)
    assert result["processed"] == 0
    assert result["output_path"] is None
    assert not output_path.exists()


def test_openai_detector_rejects_incompatible_v2_existing_output_without_overwrite(tmp_path, monkeypatch):
    input_path = _write_dataset(tmp_path, 1)
    output_path = tmp_path / "detector.json"
    output_path.write_text(json.dumps({"metadata": {}, "results": []}), encoding="utf-8")
    config = DetectorExecutionConfig(
        input_path=str(input_path),
        output_path=str(output_path),
        backend="openai",
        max_samples=1,
        overwrite=False,
    )

    monkeypatch.setattr(
        openai_agents,
        "_import_openai_agents",
        lambda: (_ for _ in ()).throw(AssertionError("invalid output should fail before SDK import")),
    )
    with pytest.raises(ValueError, match="requires v2 metadata and summary"):
        HallucinationDetectionPipeline().run_native_validation(config)
    assert json.loads(output_path.read_text()) == {"metadata": {}, "results": []}


def test_detector_resume_allows_relocated_matching_input_content(tmp_path):
    original_input = _write_dataset(tmp_path, 1, name="original.json")
    relocated_input = tmp_path / "relocated.json"
    relocated_input.write_text(original_input.read_text(encoding="utf-8"), encoding="utf-8")
    output_path = tmp_path / "detector.json"
    pipeline = HallucinationDetectionPipeline()
    original_config = DetectorExecutionConfig(
        input_path=str(original_input),
        output_path=str(output_path),
        backend="openai",
        max_samples=1,
    )
    original_plan = pipeline.build_detection_plan(original_config)
    envelope = pipeline._build_detection_envelope(
        config=original_config,
        plan=original_plan,
        results=[],
        output_path=output_path,
        grader=None,
        is_partial=True,
    )
    output_path.write_text(json.dumps(envelope), encoding="utf-8")

    relocated_config = DetectorExecutionConfig(
        input_path=str(relocated_input),
        output_path=str(output_path),
        backend="openai",
        max_samples=1,
    )
    relocated_plan = pipeline.build_detection_plan(relocated_config)

    payload, rows = pipeline._load_resume(
        output_path,
        config=relocated_config,
        plan=relocated_plan,
    )

    assert payload["metadata"]["dataset"]["sha256"] == relocated_plan["input_sha256"]
    assert rows == []


def test_generated_row_identity_persists_source_ordinal():
    row = {"prompt": "p", "response": "r", "metadata": {"model": "m", "source_ordinal": 50}}
    assert row_identity(row, 0) == "m:50"
    assert row_identity(row, 99) == "m:50"


def test_claude_detector_zero_work_does_not_import_sdk_or_write(tmp_path, monkeypatch):
    input_path = _write_dataset(tmp_path, 1)
    output_path = tmp_path / "detector.json"
    config = DetectorExecutionConfig(
        input_path=str(input_path),
        output_path=str(output_path),
        backend="claude",
        max_samples=0,
    )

    monkeypatch.setattr(
        claude_agent_sdk,
        "_import_claude_agent_sdk",
        lambda: (_ for _ in ()).throw(AssertionError("zero-work run imported SDK")),
    )
    result = HallucinationDetectionPipeline().run_native_validation(config)
    assert result["processed"] == 0
    assert result["output_path"] is None
    assert not output_path.exists()


def test_detector_modules_import_without_optional_sdks(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"agents", "claude_agent_sdk"}:
            raise AssertionError(f"optional SDK import attempted at module import time: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    for module_name in (
        "med_red_team.hallucination.detectors.base",
        "med_red_team.hallucination.detectors.openai_agents",
        "med_red_team.hallucination.detectors.claude_agent_sdk",
        "med_red_team.hallucination.grader",
        "med_red_team.hallucination.pipeline",
        "med_red_team.hallucination",
    ):
        importlib.reload(importlib.import_module(module_name))


def test_run_detector_dry_run_cli_does_not_invoke_backend(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "detector_outputs.json"
    input_path.write_text(json.dumps([{"Prompt": "p", "Response": "r", "row_idx": 0}]), encoding="utf-8")

    def fail_execute_live(config):
        raise AssertionError("dry-run must not execute detector backend")

    monkeypatch.setattr(run_detector, "_execute_live", fail_execute_live)

    assert run_detector.main(
        [
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--backend",
            "openai",
            "--json",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["planned_count"] == 1
    assert payload["plan"]["will_execute_live"] is False
    assert not output_path.exists()


def test_detector_cli_rejects_unknown_options():
    with pytest.raises(SystemExit):
        run_detector.parse_args(["--log-path", "detector.log"])


def test_explicit_cli_inputs_do_not_inherit_two_sample_example_cap(tmp_path):
    dataset = _write_dataset(tmp_path, 5)

    generation_args = run_baseline.parse_args(
        ["--dataset-path", str(dataset), "--output-dir", str(tmp_path / "generated")]
    )
    generation_config = run_baseline.load_generation_config(generation_args)
    assert generation_config.max_samples is None
    assert _build_generation_plan(generation_config)["planned_count"] == 5

    detector_args = run_detector.parse_args(
        [
            "--input-path",
            str(dataset),
            "--output-path",
            str(tmp_path / "detected.json"),
            "--backend",
            "claude",
        ]
    )
    detector_config = run_detector._load_config_from_args(detector_args)
    assert detector_config.max_samples is None
    assert detector_config.orchestrator_model == "claude-sonnet-4-20250514"
    assert detector_config.sub_agent_model == "inherit"
    assert HallucinationDetectionPipeline().build_detection_plan(detector_config)["planned_count"] == 5


def test_detector_json_live_mode_redirects_progress_to_stderr(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "detector_outputs.json"
    input_path.write_text(
        json.dumps([{"Prompt": "p", "Response": "r", "row_idx": 0}]),
        encoding="utf-8",
    )

    def fake_execute_live(config, **kwargs):
        print("provider progress")
        assert kwargs["quiet"] is True
        return {"processed": 1}

    monkeypatch.setattr(run_detector, "_execute_live", fake_execute_live)
    assert run_detector.main(
        [
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--execute-live",
            "--json",
        ]
    ) == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["result"] == {"processed": 1}
    assert "provider progress" in captured.err


def test_generation_rejects_stale_plan_before_model_construction(tmp_path, monkeypatch):
    dataset = _write_dataset(tmp_path, 1)
    config = _generation_config(dataset, tmp_path / "outputs")
    plan = _build_generation_plan(config)
    rows = _native_rows(1)
    rows[0]["Prompt"] = "changed prompt"
    dataset.write_text(json.dumps(rows), encoding="utf-8")

    class ExplodingModelPool:
        def __init__(self):
            raise AssertionError("stale plans must fail before model construction")

    with pytest.raises(ValueError, match="Generation plan is stale"):
        _generation_pipeline(model_pool_factory=ExplodingModelPool).run(
            config,
            output_path=tmp_path / "outputs" / "results.json",
            plan=plan,
            quiet=True,
        )
