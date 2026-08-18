"""Provider-independent hallucination detector interfaces and row helpers."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from med_red_team.shared.io import read_json

from ..metrics import (
    binary_classification_report,
    label_to_binary,
    merged_code_to_binary,
)
from ..schemas import OrchestratorOutput, normalize_merged_codes


SourceFormat = Literal["native", "generated"]


class DetectorOutputError(RuntimeError):
    """A row-local detector output failure that can be retried independently."""

    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        super().__init__(message)


@runtime_checkable
class HallucinationDetector(Protocol):
    """Structural interface implemented by hallucination detector backends."""

    backend: str

    def detect(
        self,
        prompt: str,
        response: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> OrchestratorOutput:
        """Classify one prompt-response pair."""
        ...


def format_detector_payload(prompt: str, response: str) -> str:
    """Format the exact prompt-response payload consumed by both detectors."""
    return (
        "<detector_input>\n"
        f"<user>{escape(prompt)}</user>\n"
        f"<llm>{escape(response)}</llm>\n"
        "</detector_input>"
    )


def extract_prompt_response(row: Any) -> tuple[str, str]:
    """Extract native uppercase or generated lowercase prompt/response fields."""
    if isinstance(row, Mapping):
        prompt = row.get("prompt", row.get("Prompt"))
        response = row.get("response", row.get("Response"))
    else:
        prompt = getattr(row, "prompt", None)
        response = getattr(row, "response", None)
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise ValueError("row must contain string Prompt/Response or prompt/response fields")
    return prompt, response


def infer_row_source_format(row: Mapping[str, Any]) -> SourceFormat:
    """Infer native versus generated semantics from one source row."""
    if "Prompt" in row and "Response" in row:
        return "native"
    if "prompt" in row and "response" in row:
        return "generated"
    raise ValueError("row must contain Prompt/Response or prompt/response fields")


def row_identity(
    row: Mapping[str, Any],
    ordinal: int | None = None,
    *,
    source_format: SourceFormat | None = None,
) -> str:
    """Return row_idx for native rows and model+source ordinal for generated rows."""
    resolved_format = source_format or infer_row_source_format(row)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}

    if resolved_format == "native":
        row_idx = row.get("row_idx", metadata.get("row_idx"))
        if row_idx is None:
            raise ValueError("native row lacks row_idx identity")
        return str(row_idx)

    model = metadata.get("model") or row.get("model") or row.get("Model")
    if not model:
        raise ValueError("generated row lacks model identity")
    source_ordinal = metadata.get("source_ordinal", ordinal)
    if source_ordinal is None:
        raise ValueError("ordinal is required for generated rows without metadata.source_ordinal")
    return f"{model}:{source_ordinal}"


def coerce_orchestrator_output(value: Any) -> OrchestratorOutput:
    """Validate provider output as the shared structured schema."""
    if isinstance(value, OrchestratorOutput):
        return value
    if isinstance(value, Mapping):
        return OrchestratorOutput.model_validate(dict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return OrchestratorOutput.model_validate(model_dump(mode="json"))
    raise ValueError(f"Invalid orchestrator output type: {type(value).__name__}")


def merge_orchestrator_output_into_row(
    row: Mapping[str, Any],
    output: OrchestratorOutput,
) -> dict[str, Any]:
    """Preserve a source row and append normalized detector fields."""
    out = dict(row)
    out.update(
        {
            "merged_codes": output.merged_codes,
            "rationale": output.rationale,
            "agent_decisions": [
                decision.model_dump(mode="json", exclude_none=True)
                for decision in output.agent_decisions
            ],
        }
    )
    return out


def read_input_rows(path: str | Path) -> tuple[list[dict[str, Any]], SourceFormat]:
    """Read native list JSON or legacy/v2 generated-response envelopes."""
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return [dict(row) for row in payload["results"]], "generated"
    if isinstance(payload, list):
        return [dict(row) for row in payload], "native"
    raise ValueError(f"Unrecognized detector input JSON format: {path}")


def summarize_detector_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return metrics over successful rows and explicit detector-failure coverage."""
    successful_rows = [
        row
        for row in rows
        if not row.get("detection_error") and "merged_codes" in row
    ]
    failed_rows = [
        row
        for row in rows
        if row.get("detection_error") or "merged_codes" not in row
    ]
    total = len(successful_rows)
    positives = sum(
        merged_code_to_binary(row["merged_codes"]) or 0
        for row in successful_rows
    )
    summary: dict[str, Any] = {
        "result_count": len(rows),
        "successful_count": len(successful_rows),
        "failed_count": len(failed_rows),
        "metrics_denominator": total,
        "unlabeled_summary": {
            "total": total,
            "predicted_positive": positives,
            "predicted_negative": total - positives,
            "positive_rate": positives / total if total else 0.0,
        },
    }

    root_counts = {str(code): 0 for code in range(1, 8)}
    uncertain = 0
    for row in successful_rows:
        merged = normalize_merged_codes(row["merged_codes"])
        if merged == "0.5":
            uncertain += 1
        elif isinstance(merged, list):
            for code in set(merged):
                root_counts[code] += 1
    summary["unlabeled_summary"].update(
        {"root_code_counts": root_counts, "uncertain_count": uncertain}
    )

    labeled_rows = []
    for row in successful_rows:
        if "Hallucination/Accuracy" not in row:
            continue
        try:
            if label_to_binary(row.get("Hallucination/Accuracy")) is not None:
                labeled_rows.append(row)
        except ValueError:
            continue
    if labeled_rows:
        y_true = [row.get("Hallucination/Accuracy") for row in labeled_rows]
        y_pred = [row["merged_codes"] for row in labeled_rows]
        summary["labeled_native_metrics"] = binary_classification_report(y_true, y_pred)
    return summary


__all__ = [
    "DetectorOutputError",
    "HallucinationDetector",
    "SourceFormat",
    "coerce_orchestrator_output",
    "extract_prompt_response",
    "format_detector_payload",
    "infer_row_source_format",
    "merge_orchestrator_output_into_row",
    "read_input_rows",
    "row_identity",
    "summarize_detector_rows",
]
