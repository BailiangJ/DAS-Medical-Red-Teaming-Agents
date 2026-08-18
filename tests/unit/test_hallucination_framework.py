from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from med_red_team.hallucination.config import DetectorExecutionConfig
from med_red_team.hallucination.data import (
    load_detector_outputs,
    load_prompt_responses,
)
from med_red_team.hallucination.detectors import HallucinationDetector, openai_agents
from med_red_team.hallucination.detectors.audit import (
    attach_execution_audit,
    observed_specialist_result,
)
from med_red_team.hallucination.detectors.base import (
    DetectorOutputError,
    format_detector_payload,
    row_identity,
)
from med_red_team.hallucination.detectors import claude_agent_sdk
from med_red_team.hallucination.detectors.claude_agent_sdk import (
    ClaudeAgentSDKHallucinationDetector,
)
from med_red_team.hallucination.grader import HallucinationGrader
from med_red_team.hallucination.pipeline import HallucinationDetectionPipeline
from med_red_team.hallucination.schemas import (
    OrchestratorOutput,
    SpecialistExecutionAudit,
)
from med_red_team.shared.checkpoints import checkpoint_path
from med_red_team.shared.io import read_json


def _output(code: str = "0") -> OrchestratorOutput:
    decisions: list[dict[str, Any]] = []
    for index in range(1, 8):
        if index == 1 and code == "0.5":
            decisions.append(
                {
                    "code": index,
                    "called": True,
                    "reasoning": "uncertain",
                    "classification": "0.5",
                    "cls_reasoning": "uncertain",
                }
            )
        elif index == 1 and code == "1":
            decisions.append(
                {
                    "code": index,
                    "called": True,
                    "reasoning": "fact issue",
                    "classification": ["1A"],
                    "cls_reasoning": "false claim",
                }
            )
        else:
            decisions.append(
                {
                    "code": index,
                    "called": False,
                    "reasoning": "not needed",
                }
            )
    merged_codes: str | list[str] = ["1"] if code == "1" else code
    return OrchestratorOutput(
        merged_codes=merged_codes,
        rationale=f"result {code}",
        agent_decisions=decisions,
    )


class _FakeDetector:
    backend = "fake"
    detector_role = "test"

    def __init__(
        self,
        codes: list[str] | None = None,
        *,
        fail_on_call: int | None = None,
    ) -> None:
        self.codes = list(codes or ["0"])
        self.fail_on_call = fail_on_call
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def detect(
        self,
        prompt: str,
        response: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> OrchestratorOutput:
        self.calls.append((prompt, response, dict(metadata or {})))
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("detector stopped")
        code = self.codes[min(len(self.calls) - 1, len(self.codes) - 1)]
        return _output(code)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "backend": self.backend,
            "role": self.detector_role,
        }


def _native_rows(count: int) -> list[dict[str, Any]]:
    return [
        {
            "Prompt": f"prompt {index}",
            "Response": f"response {index}",
            "row_idx": index,
            "Hallucination/Accuracy": 1,
        }
        for index in range(count)
    ]


def test_hallucination_grader_protocol_and_tri_state_results():
    detector = _FakeDetector(["0", "0.5", "1"])
    assert isinstance(detector, HallucinationDetector)
    grader = HallucinationGrader(detector)

    negative = grader.grade_prompt_response("p0", "r0")
    uncertain = grader.grade_prompt_response("p1", "r1")
    positive = grader.grade({"Prompt": "p2"}, "r2", {})

    assert (negative.score, negative.is_correct, negative.hallucination_detected) == (
        0.0,
        True,
        False,
    )
    assert (uncertain.score, uncertain.is_correct, uncertain.hallucination_detected) == (
        0.5,
        None,
        None,
    )
    assert (positive.score, positive.is_correct, positive.hallucination_detected) == (
        1.0,
        False,
        True,
    )
    assert positive.merged_codes == ["1"]
    assert grader.to_dict()["detector"]["backend"] == "fake"


def test_detection_pipeline_writes_native_and_generated_v2_envelopes(tmp_path):
    native_path = tmp_path / "native.json"
    native_path.write_text(json.dumps(_native_rows(2)), encoding="utf-8")
    native_output = tmp_path / "native_output.json"
    native_config = DetectorExecutionConfig(
        input_path=str(native_path),
        output_path=str(native_output),
        backend="openai",
    )
    detector = _FakeDetector(["1", "0"])
    pipeline = HallucinationDetectionPipeline(HallucinationGrader(detector))

    native_result = pipeline.run_native_validation(
        native_config,
        quiet=True,
        save_every=0,
    )
    native_payload = read_json(native_output)

    assert native_result["processed"] == 2
    assert len(detector.calls) == 2
    assert native_payload["metadata"]["phase"] == "native_validation"
    assert native_payload["metadata"]["dataset"]["source_format"] == "native"
    assert native_payload["results"][0]["metadata"]["source_identity"] == "0"
    assert native_payload["summary"]["labeled_native_metrics"]["counts"]["total"] == 2

    generated_path = tmp_path / "generated.json"
    generated_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "schema_version": "2.0",
                    "axis": "hallucination",
                    "phase": "response_generation",
                    "models": {"target_model": "fake-model"},
                    "config": {},
                    "dataset": {"planned_count": 1},
                    "source": {"fixture": "generated"},
                    "is_partial": False,
                },
                "summary": {"total_responses": 1},
                "results": [
                    {
                        "prompt": "generated prompt",
                        "response": "generated response",
                        "raw_response": "raw response",
                        "reasoning": "source reasoning",
                        "row_idx": 12,
                        "sample_id": "12",
                        "metadata": {"model": "fake-model"},
                        "error": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    generated_output = tmp_path / "generated_output.json"
    generated_config = DetectorExecutionConfig(
        input_path=str(generated_path),
        output_path=str(generated_output),
        backend="openai",
        input_format="generated",
    )
    generated_detector = _FakeDetector()
    generated_pipeline = HallucinationDetectionPipeline(
        HallucinationGrader(generated_detector)
    )

    generated_pipeline.run_generated_response_detection(
        generated_config,
        quiet=True,
        save_every=0,
    )
    generated_payload = read_json(generated_output)

    assert generated_payload["metadata"]["phase"] == "generated_response_detection"
    row = generated_payload["results"][0]
    assert row["raw_response"] == "raw response"
    assert row["reasoning"] == "source reasoning"
    assert row["metadata"]["source_identity"] == "fake-model:12"
    assert row["metadata"]["source_ordinal"] == 12

    response_envelope = load_prompt_responses(
        generated_path,
        source_format="generated",
    )
    assert response_envelope.schema_version == "2.0"
    assert response_envelope.responses[0].source_row["raw_response"] == "raw response"
    detector_envelope = load_detector_outputs(generated_output)
    assert detector_envelope.schema_version == "2.0"
    assert detector_envelope.records[0].identity == "fake-model:12"


def test_detection_zero_work_and_complete_resume_avoid_detector_construction(tmp_path):
    input_path = tmp_path / "native.json"
    input_path.write_text(json.dumps(_native_rows(1)), encoding="utf-8")
    output_path = tmp_path / "output.json"

    def exploding_factory(config):
        raise AssertionError("detector factory must not be called")

    zero_config = DetectorExecutionConfig(
        input_path=str(input_path),
        output_path=str(output_path),
        max_samples=0,
    )
    zero_pipeline = HallucinationDetectionPipeline(grader_factory=exploding_factory)
    zero_result = zero_pipeline.run_native_validation(zero_config, quiet=True)
    assert zero_result["processed"] == 0
    assert not output_path.exists()

    config = DetectorExecutionConfig(
        input_path=str(input_path),
        output_path=str(output_path),
    )
    HallucinationDetectionPipeline(HallucinationGrader(_FakeDetector())).run_native_validation(
        config,
        quiet=True,
        save_every=0,
    )
    original = output_path.read_bytes()

    complete_pipeline = HallucinationDetectionPipeline(grader_factory=exploding_factory)
    result = complete_pipeline.run_native_validation(config, quiet=True)
    assert result["processed"] == 0
    assert "already complete" in result["message"]
    assert output_path.read_bytes() == original


def test_detection_checkpoint_resume_uses_stable_native_identity(tmp_path):
    input_path = tmp_path / "native.json"
    input_path.write_text(json.dumps(_native_rows(3)), encoding="utf-8")
    output_path = tmp_path / "output.json"
    config = DetectorExecutionConfig(
        input_path=str(input_path),
        output_path=str(output_path),
    )

    failing_detector = _FakeDetector(fail_on_call=2)
    failing_pipeline = HallucinationDetectionPipeline(
        HallucinationGrader(failing_detector)
    )
    with pytest.raises(RuntimeError, match="detector stopped"):
        failing_pipeline.run_native_validation(
            config,
            quiet=True,
            save_every=1,
        )

    partial_path = checkpoint_path(output_path)
    partial = read_json(partial_path)
    assert partial["metadata"]["is_partial"] is True
    assert [row["metadata"]["source_identity"] for row in partial["results"]] == ["0"]

    resumed_detector = _FakeDetector()
    resumed_pipeline = HallucinationDetectionPipeline(
        HallucinationGrader(resumed_detector)
    )
    result = resumed_pipeline.run_native_validation(
        config,
        quiet=True,
        save_every=0,
    )

    assert result["processed"] == 2
    assert [call[0] for call in resumed_detector.calls] == ["prompt 1", "prompt 2"]
    assert [row["metadata"]["source_identity"] for row in read_json(output_path)["results"]] == [
        "0",
        "1",
        "2",
    ]
    assert not partial_path.exists()


def test_detection_rejects_stale_plan_before_grader_construction(tmp_path):
    input_path = tmp_path / "native.json"
    input_path.write_text(json.dumps(_native_rows(1)), encoding="utf-8")
    config = DetectorExecutionConfig(
        input_path=str(input_path),
        output_path=str(tmp_path / "output.json"),
    )
    pipeline = HallucinationDetectionPipeline(
        grader_factory=lambda config: (_ for _ in ()).throw(
            AssertionError("stale plans must fail before grader construction")
        )
    )
    plan = pipeline.build_detection_plan(config)
    changed = _native_rows(1)
    changed[0]["Prompt"] = "changed prompt"
    input_path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="Detection plan is stale"):
        pipeline.run_native_validation(config, plan=plan, quiet=True)


def test_ignore_existing_failure_checkpoint_contains_only_regenerated_rows(tmp_path):
    input_path = tmp_path / "native.json"
    input_path.write_text(json.dumps(_native_rows(3)), encoding="utf-8")
    output_path = tmp_path / "output.json"
    initial_config = DetectorExecutionConfig(
        input_path=str(input_path),
        output_path=str(output_path),
    )
    HallucinationDetectionPipeline(HallucinationGrader(_FakeDetector())).run_native_validation(
        initial_config,
        quiet=True,
        save_every=0,
    )

    rerun_config = DetectorExecutionConfig(
        input_path=str(input_path),
        output_path=str(output_path),
        ignore_existing=True,
    )
    failing = HallucinationDetectionPipeline(
        HallucinationGrader(_FakeDetector(["1"], fail_on_call=2))
    )
    with pytest.raises(RuntimeError, match="detector stopped"):
        failing.run_native_validation(rerun_config, quiet=True, save_every=1)

    partial_path = checkpoint_path(output_path)
    partial = read_json(partial_path)
    assert [row["metadata"]["source_identity"] for row in partial["results"]] == ["0"]
    assert partial["results"][0]["merged_codes"] == ["1"]

    compatible = DetectorExecutionConfig(
        input_path=str(input_path),
        output_path=str(output_path),
        ignore_existing=False,
    )
    authoritative_detector = _FakeDetector()
    authoritative = HallucinationDetectionPipeline(
        HallucinationGrader(authoritative_detector)
    ).run_native_validation(compatible, quiet=True)
    assert authoritative["processed"] == 0
    assert authoritative_detector.calls == []

    resumed_output = tmp_path / "resumed_output.json"
    resumed_config = DetectorExecutionConfig(
        input_path=str(input_path),
        output_path=str(resumed_output),
        ignore_existing=True,
    )
    resumed_detector = _FakeDetector()
    result = HallucinationDetectionPipeline(
        HallucinationGrader(resumed_detector)
    ).run_native_validation(
        resumed_config,
        resume_path=partial_path,
        quiet=True,
        save_every=0,
    )
    assert result["processed"] == 2
    assert [call[0] for call in resumed_detector.calls] == ["prompt 1", "prompt 2"]
    final = read_json(resumed_output)
    assert final["results"][0]["merged_codes"] == ["1"]
    assert not checkpoint_path(resumed_output).exists()


def test_generated_identity_requires_model_name():
    with pytest.raises(ValueError, match="lacks model identity"):
        row_identity(
            {"prompt": "p", "response": "r", "metadata": {"source_ordinal": 0}},
            source_format="generated",
        )


def test_claude_sync_detector_works_inside_running_event_loop(monkeypatch):
    detector = ClaudeAgentSDKHallucinationDetector(orchestrator_options=object())

    async def fake_adetect(prompt, response, *, metadata=None):
        return _output("0")

    monkeypatch.setattr(detector, "adetect", fake_adetect)

    async def invoke():
        return detector.detect("prompt", "response")

    output = asyncio.run(invoke())
    assert output.merged_codes == "0"


def test_claude_query_prefers_terminal_result_and_concatenates_fallback_blocks(monkeypatch):
    class TextBlock:
        def __init__(self, text):
            self.text = text

    class AssistantMessage:
        def __init__(self, content):
            self.content = content
            self.parent_tool_use_id = None

    class ResultMessage:
        def __init__(self, result):
            self.result = result
            self.structured_output = None
            self.is_error = False

    terminal_json = _output("1").model_dump_json()

    async def terminal_query(**kwargs):
        yield AssistantMessage([TextBlock("not json")])
        yield ResultMessage(terminal_json)

    monkeypatch.setattr(
        claude_agent_sdk,
        "_import_claude_agent_sdk",
        lambda: (object, AssistantMessage, object, TextBlock, terminal_query),
    )
    terminal_output = asyncio.run(
        claude_agent_sdk._query_output("prompt", "response", object())
    )
    assert terminal_output.merged_codes == ["1"]

    split_json = _output("0").model_dump_json()

    async def split_query(**kwargs):
        midpoint = len(split_json) // 2
        yield AssistantMessage(
            [TextBlock(split_json[:midpoint]), TextBlock(split_json[midpoint:])]
        )

    monkeypatch.setattr(
        claude_agent_sdk,
        "_import_claude_agent_sdk",
        lambda: (object, AssistantMessage, object, TextBlock, split_query),
    )
    split_output = asyncio.run(
        claude_agent_sdk._query_output("prompt", "response", object())
    )
    assert split_output.merged_codes == "0"


def test_execution_audit_warns_without_rewriting_merged_codes():
    reported = _output("1")
    output = OrchestratorOutput(
        merged_codes="0",
        rationale=reported.rationale,
        agent_decisions=reported.agent_decisions,
    )
    observed = [
        SpecialistExecutionAudit(
            name="MedFactChecker",
            status="completed",
            normalized_codes=["1A"],
        )
    ]

    with pytest.warns(RuntimeWarning, match="execution metadata is inconsistent"):
        audit = attach_execution_audit(output, observed)

    assert output.merged_codes == "0"
    assert "decision_merge_mismatch" in audit.messages
    assert "observed_merge_mismatch" in audit.messages
    assert "prompt" not in json.dumps(audit.model_dump(mode="json"))


def test_list_form_tool_result_is_parsed_for_execution_audit():
    entry = observed_specialist_result(
        "MedFactChecker",
        [{"type": "text", "text": '{"classification":["1A"]}'}],
    )

    assert entry is not None
    assert entry.status == "completed"
    assert entry.normalized_codes == ["1A"]


def test_openai_detector_collects_compact_tool_execution_audit(monkeypatch):
    call = type(
        "ToolCallItem",
        (),
        {"tool_name": "MedFactChecker", "call_id": "call-1"},
    )()
    result_item = type(
        "ToolCallOutputItem",
        (),
        {"call_id": "call-1", "output": {"classification": ["1A"]}},
    )()
    run_result = type(
        "RunResult",
        (),
        {"final_output": _output("1"), "new_items": [call, result_item]},
    )()

    class Runner:
        @staticmethod
        def run_sync(*args, **kwargs):
            return run_result

    monkeypatch.setattr(
        openai_agents,
        "_import_openai_agents",
        lambda: (object, object, Runner, object),
    )
    detector = openai_agents.OpenAIAgentsHallucinationDetector(
        search_agent_model="other",
        orchestrator=object(),
    )

    output = detector.detect("secret prompt", "secret response")

    assert output.execution_audit is not None
    assert output.execution_audit.status == "available"
    serialized = json.dumps(output.execution_audit.model_dump(mode="json"))
    assert "secret prompt" not in serialized
    assert "secret response" not in serialized
    assert output.merged_codes == ["1"]


def test_execution_audit_persists_through_detection_envelope(tmp_path):
    input_path = tmp_path / "native.json"
    input_path.write_text(json.dumps(_native_rows(1)), encoding="utf-8")
    output_path = tmp_path / "output.json"

    class AuditedDetector(_FakeDetector):
        def detect(self, prompt, response, *, metadata=None):
            output = _output("1")
            attach_execution_audit(
                output,
                [
                    SpecialistExecutionAudit(
                        name="MedFactChecker",
                        status="completed",
                        normalized_codes=["1A"],
                    )
                ],
            )
            return output

    config = DetectorExecutionConfig(
        input_path=str(input_path),
        output_path=str(output_path),
    )
    HallucinationDetectionPipeline(
        HallucinationGrader(AuditedDetector())
    ).run_native_validation(config, quiet=True, save_every=0)

    payload = read_json(output_path)
    assert payload["results"][0]["execution_audit"]["status"] == "available"
    loaded = load_detector_outputs(output_path)
    assert loaded.records[0].execution_audit["status"] == "available"


def test_detector_payload_preserves_whitespace_and_escapes_boundaries():
    payload = format_detector_payload("  <user>x</user>  ", "<&>")

    assert "  &lt;user&gt;x&lt;/user&gt;  " in payload
    assert "&lt;&amp;&gt;" in payload
    assert payload.startswith("<detector_input>")


def test_row_local_detector_error_continues_and_is_excluded_from_metrics(tmp_path):
    input_path = tmp_path / "native.json"
    input_path.write_text(json.dumps(_native_rows(2)), encoding="utf-8")
    output_path = tmp_path / "output.json"

    class RowLocalDetector(_FakeDetector):
        def detect(self, prompt, response, *, metadata=None):
            self.calls.append((prompt, response, metadata))
            if len(self.calls) == 1:
                raise DetectorOutputError(
                    "invalid_orchestrator_output",
                    "invalid detector output",
                )
            return _output("1")

    detector = RowLocalDetector()
    config = DetectorExecutionConfig(
        input_path=str(input_path),
        output_path=str(output_path),
    )
    result = HallucinationDetectionPipeline(
        HallucinationGrader(detector)
    ).run_native_validation(config, quiet=True, save_every=0)

    payload = read_json(output_path)
    assert result["processed"] == 2
    assert len(detector.calls) == 2
    assert payload["results"][0]["detection_error"]["retryable"] is True
    assert payload["summary"]["result_count"] == 2
    assert payload["summary"]["successful_count"] == 1
    assert payload["summary"]["failed_count"] == 1
    assert payload["summary"]["metrics_denominator"] == 1


def test_detector_error_rows_are_retried_on_rerun(tmp_path):
    input_path = tmp_path / "native.json"
    input_path.write_text(json.dumps(_native_rows(2)), encoding="utf-8")
    output_path = tmp_path / "output.json"

    class FirstRunDetector(_FakeDetector):
        def detect(self, prompt, response, *, metadata=None):
            self.calls.append((prompt, response, metadata))
            if len(self.calls) == 1:
                raise DetectorOutputError("invalid_json", "retry")
            return _output("0")

    config = DetectorExecutionConfig(
        input_path=str(input_path),
        output_path=str(output_path),
    )
    HallucinationDetectionPipeline(
        HallucinationGrader(FirstRunDetector())
    ).run_native_validation(config, quiet=True, save_every=0)

    retry_detector = _FakeDetector(["1"])
    result = HallucinationDetectionPipeline(
        HallucinationGrader(retry_detector)
    ).run_native_validation(config, quiet=True, save_every=0)

    payload = read_json(output_path)
    assert result["processed"] == 1
    assert [call[0] for call in retry_detector.calls] == ["prompt 0"]
    assert payload["summary"]["successful_count"] == 2
    assert payload["summary"]["failed_count"] == 0


def test_improved_checkpoint_wins_over_stale_failed_final_output(tmp_path):
    input_path = tmp_path / "native.json"
    input_path.write_text(json.dumps(_native_rows(3)), encoding="utf-8")
    output_path = tmp_path / "output.json"
    config = DetectorExecutionConfig(
        input_path=str(input_path),
        output_path=str(output_path),
    )

    class InitialDetector(_FakeDetector):
        def detect(self, prompt, response, *, metadata=None):
            self.calls.append((prompt, response, metadata))
            if len(self.calls) <= 2:
                raise DetectorOutputError("invalid_json", "retry")
            return _output("0")

    HallucinationDetectionPipeline(
        HallucinationGrader(InitialDetector())
    ).run_native_validation(config, quiet=True, save_every=0)

    class InterruptedRetry(_FakeDetector):
        def detect(self, prompt, response, *, metadata=None):
            self.calls.append((prompt, response, metadata))
            if len(self.calls) == 2:
                raise RuntimeError("provider stopped")
            return _output("1")

    with pytest.raises(RuntimeError, match="provider stopped"):
        HallucinationDetectionPipeline(
            HallucinationGrader(InterruptedRetry())
        ).run_native_validation(config, quiet=True, save_every=1)

    checkpoint = read_json(checkpoint_path(output_path))
    assert checkpoint["summary"]["successful_count"] == 2
    final_before = read_json(output_path)
    assert final_before["summary"]["successful_count"] == 1

    final_retry = _FakeDetector(["1"])
    result = HallucinationDetectionPipeline(
        HallucinationGrader(final_retry)
    ).run_native_validation(config, quiet=True, save_every=0)

    assert result["processed"] == 1
    assert [call[0] for call in final_retry.calls] == ["prompt 1"]
    assert read_json(output_path)["summary"]["successful_count"] == 3


def test_custom_key_native_output_reloads_and_resumes(tmp_path):
    input_path = tmp_path / "custom.json"
    input_path.write_text(
        json.dumps([
            {
                "question": "p",
                "answer": "r",
                "id": "a",
                "Prompt": "legacy prompt",
                "prompt": "unrelated prompt",
                "Hallucination/Accuracy": "positive",
            }
        ]),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.json"
    config = DetectorExecutionConfig(
        input_path=str(input_path),
        output_path=str(output_path),
        prompt_key="question",
        response_key="answer",
        row_id_key="id",
    )
    first = _FakeDetector()
    HallucinationDetectionPipeline(
        HallucinationGrader(first)
    ).run_native_validation(config, quiet=True, save_every=0)

    second = _FakeDetector()
    result = HallucinationDetectionPipeline(
        HallucinationGrader(second)
    ).run_native_validation(config, quiet=True, save_every=0)
    loaded = load_detector_outputs(output_path, source_format="native")

    assert result["processed"] == 0
    assert second.calls == []
    assert loaded.records[0].identity == "a"
    assert loaded.records[0].prompt == "p"
    assert loaded.records[0].response == "r"


def test_partial_generated_input_uses_only_valid_available_rows(tmp_path):
    input_path = tmp_path / "partial.json"
    input_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "schema_version": "2.0",
                    "axis": "hallucination",
                    "phase": "response_generation",
                    "config": {},
                    "models": {"target_model": "model"},
                    "dataset": {"planned_count": 5},
                    "source": {"fixture": "partial"},
                    "is_partial": True,
                },
                "summary": {"planned_count": 5},
                "results": [
                    {
                        "prompt": "valid",
                        "response": "answer",
                        "sample_id": "a",
                        "metadata": {"model": "model"},
                        "error": None,
                    },
                    {
                        "prompt": "failed",
                        "response": "",
                        "sample_id": "b",
                        "metadata": {"model": "model"},
                        "error": {"type": "Error", "message": "failed"},
                    },
                    {
                        "prompt": "blank",
                        "response": "   ",
                        "sample_id": "c",
                        "metadata": {"model": "model"},
                        "error": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "detected.json"
    detector = _FakeDetector()
    config = DetectorExecutionConfig(
        input_path=str(input_path),
        output_path=str(output_path),
        input_format="generated",
    )

    plan = HallucinationDetectionPipeline().build_detection_plan(config)
    result = HallucinationDetectionPipeline(
        HallucinationGrader(detector)
    ).run_generated_response_detection(config, plan=plan, quiet=True, save_every=0)

    assert plan["source_planned_count"] == 5
    assert plan["source_available_count"] == 3
    assert plan["eligible_input_rows"] == 1
    assert plan["excluded_by_reason"] == {
        "generation_error": 1,
        "blank_response": 1,
    }
    assert result["processed"] == 1
    assert len(detector.calls) == 1
    assert result["summary"]["source_is_partial"] is True
