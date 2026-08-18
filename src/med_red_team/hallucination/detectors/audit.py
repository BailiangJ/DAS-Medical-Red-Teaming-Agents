"""Compact provider execution metadata for Hallucination specialists."""

from __future__ import annotations

from collections.abc import Mapping
import json
import re
from typing import Any
import warnings

from ..schemas import (
    ExecutionAudit,
    OrchestratorOutput,
    SpecialistExecutionAudit,
    merge_agent_classifications,
    normalize_classification,
    normalize_merged_codes,
)


SPECIALIST_NAME_TO_CODE = {
    "MedFactChecker": 1,
    "CitationVerifier": 2,
    "ReasoningAuditor": 3,
    "ContextKeeper": 4,
    "SafetyGuardian": 5,
    "InstructionWatcher": 6,
    "HallucinationScout": 7,
}
CODE_TO_SPECIALIST_NAME = {
    code: name for name, code in SPECIALIST_NAME_TO_CODE.items()
}


def _classification_tokens(value: Any) -> list[str]:
    normalized = normalize_classification(value)
    if isinstance(normalized, str):
        return [normalized]
    return [str(code) for code in normalized]


def _mapping_from_value(value: Any) -> dict[str, Any] | None:
    if isinstance(value, list):
        text_parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping) and "classification" in item:
                return dict(item)
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        value = "".join(text_parts) if text_parts else None
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        return dict(dumped) if isinstance(dumped, Mapping) else None
    if isinstance(value, str):
        text = value.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        if fenced:
            text = fenced.group(1)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return dict(parsed) if isinstance(parsed, Mapping) else None
    return None


def observed_specialist_result(
    name: str,
    value: Any = None,
    *,
    failed: bool = False,
) -> SpecialistExecutionAudit | None:
    """Convert one allowlisted provider tool result into a compact audit entry."""
    if name not in SPECIALIST_NAME_TO_CODE:
        return None
    if failed:
        return SpecialistExecutionAudit(name=name, status="failed")
    mapping = _mapping_from_value(value)
    if mapping is None or "classification" not in mapping:
        return SpecialistExecutionAudit(name=name, status="called_no_result")
    try:
        codes = _classification_tokens(mapping["classification"])
    except (TypeError, ValueError):
        return SpecialistExecutionAudit(name=name, status="called_no_result")
    return SpecialistExecutionAudit(
        name=name,
        status="completed",
        normalized_codes=codes,
    )


def _roots_from_tokens(tokens: list[str]) -> list[str]:
    roots = {
        token[0]
        for token in tokens
        if token not in {"0", "0.5"} and token and token[0] in "1234567"
    }
    return sorted(roots, key=int)


def attach_execution_audit(
    output: OrchestratorOutput,
    observed: list[SpecialistExecutionAudit] | None,
) -> ExecutionAudit:
    """Attach compact provider execution metadata to the detector output."""
    messages: list[str] = []
    decision_merge = normalize_merged_codes(
        merge_agent_classifications(output.agent_decisions)
    )
    reported_merge = normalize_merged_codes(output.merged_codes)
    if decision_merge != reported_merge:
        messages.append("decision_merge_mismatch")

    if observed is None:
        audit = ExecutionAudit(
            status="unavailable",
            specialists=[],
            messages=messages,
        )
    else:
        observed_names = {
            entry.name
            for entry in observed
            if entry.status in {"completed", "called_no_result", "failed"}
        }
        reported_names = {
            CODE_TO_SPECIALIST_NAME[decision.code]
            for decision in output.agent_decisions
            if decision.called
        }
        if observed_names != reported_names:
            messages.append("observed_calls_mismatch")

        observed_by_name: dict[str, list[str]] = {}
        for entry in observed:
            if entry.status == "completed":
                observed_by_name.setdefault(entry.name, []).extend(
                    entry.normalized_codes
                )
        for decision in output.agent_decisions:
            name = CODE_TO_SPECIALIST_NAME[decision.code]
            if name not in observed_by_name or decision.classification is None:
                continue
            try:
                reported_codes = set(
                    _classification_tokens(decision.classification)
                )
            except (TypeError, ValueError):
                messages.append("observed_codes_mismatch")
                break
            if set(observed_by_name[name]) != reported_codes:
                messages.append("observed_codes_mismatch")
                break

        observed_roots = _roots_from_tokens(
            [
                token
                for codes in observed_by_name.values()
                for token in codes
            ]
        )
        if observed_roots:
            normalized_observed: str | list[str] = observed_roots
            if normalized_observed != reported_merge:
                messages.append("observed_merge_mismatch")

        status = "available" if all(
            entry.status == "completed" for entry in observed
        ) else "partial"
        audit = ExecutionAudit(
            status=status,
            specialists=observed,
            messages=list(dict.fromkeys(messages)),
        )

    output.attach_execution_audit(audit)
    if audit.messages:
        warnings.warn(
            "Detector execution metadata is inconsistent; inspect execution_audit: "
            + ", ".join(audit.messages),
            RuntimeWarning,
            stacklevel=2,
        )
    return audit


__all__ = [
    "CODE_TO_SPECIALIST_NAME",
    "SPECIALIST_NAME_TO_CODE",
    "attach_execution_audit",
    "observed_specialist_result",
]
