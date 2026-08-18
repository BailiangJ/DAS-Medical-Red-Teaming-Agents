"""Typed loaders for hallucination prompt/response and detector artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, TypeVar

from med_red_team.shared.io import read_json, to_jsonable, validate_result_metadata

from .config import resolve_max_samples
from .schemas import normalize_merged_codes


SourceFormat = Literal["native", "generated"]
RecordT = TypeVar("RecordT")


class HallucinationDataError(ValueError):
    """Raised when a hallucination artifact has an unexpected shape."""


@dataclass(frozen=True)
class PromptResponseRecord:
    """Native Prompt/Response row with row_idx identity."""

    identity: str
    prompt: str
    response: str
    row_idx: int | str
    model: str | None = None
    label: float | None = None
    additional_comments: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_row: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None
    source_format: SourceFormat = "native"


@dataclass(frozen=True)
class GeneratedResponseRecord:
    """Legacy generated response row with model+ordinal identity."""

    identity: str
    prompt: str
    response: Any
    model: str
    ordinal: int | None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Any = None
    source_row: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None
    source_format: SourceFormat = "generated"


@dataclass(frozen=True)
class GeneratedResponseEnvelope:
    """Legacy or v2 generated response envelope."""

    path: str
    statistics: dict[str, Any]
    responses: list[GeneratedResponseRecord]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str | None = None


@dataclass(frozen=True)
class DetectorRecord:
    """Detector output row for native or generated prompt/response pairs."""

    identity: str
    prompt: str
    response: str
    merged_codes: str | list[str] | None
    rationale: str | None
    agent_decisions: list[dict[str, Any]]
    execution_audit: dict[str, Any] | None = None
    row_idx: int | str | None = None
    model: str | None = None
    ordinal: int | None = None
    label: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Any = None
    source_row: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None
    source_format: SourceFormat = "native"


@dataclass(frozen=True)
class DetectorOutputList:
    """A loaded legacy list or v2 detector envelope."""

    path: str
    source_format: SourceFormat
    records: list[DetectorRecord]
    metadata: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    schema_version: str | None = None


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()
    file_path = Path(path)
    try:
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        raise HallucinationDataError(f"Unable to read file for SHA256: {file_path}") from exc
    return digest.hexdigest()


def dataset_sha256(path: str | Path) -> str:
    """Alias for file-level dataset SHA256 checks."""
    return sha256_file(path)


def sha256_json_value(value: Any) -> str:
    """Return a deterministic SHA256 digest for a JSON-serializable value."""
    payload = json.dumps(
        to_jsonable(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: str | Path) -> Any:
    file_path = Path(path)
    try:
        return read_json(file_path)
    except FileNotFoundError as exc:
        raise HallucinationDataError(f"File not found: {file_path}") from exc
    except json.JSONDecodeError as exc:
        raise HallucinationDataError(f"Invalid JSON in {file_path}: {exc}") from exc
    except OSError as exc:
        raise HallucinationDataError(f"Unable to read {file_path}: {exc}") from exc


def _require_mapping(value: Any, *, path: Path, row: int | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        where = f" row {row}" if row is not None else ""
        raise HallucinationDataError(f"Expected object at {path}{where}, got {type(value).__name__}")
    return value


def _require_list(value: Any, *, path: Path, name: str = "top-level JSON") -> list[Any]:
    if not isinstance(value, list):
        raise HallucinationDataError(f"Expected {name} in {path} to be a list, got {type(value).__name__}")
    return value


def _require_key(row: dict[str, Any], key: str, *, path: Path, index: int) -> Any:
    if key not in row:
        raise HallucinationDataError(f"Missing required key {key!r} in {path} row {index}")
    return row[key]


def _require_text(row: dict[str, Any], key: str, *, path: Path, index: int) -> str:
    value = _require_key(row, key, path=path, index=index)
    if value is None:
        raise HallucinationDataError(f"Key {key!r} cannot be null in {path} row {index}")
    if not isinstance(value, str):
        raise HallucinationDataError(f"Key {key!r} must be a string in {path} row {index}")
    return value


def _validate_v2_envelope_metadata(
    metadata: dict[str, Any],
    *,
    path: Path,
    expected_phase: str | set[str],
) -> None:
    try:
        validate_result_metadata(metadata)
    except (TypeError, ValueError) as exc:
        raise HallucinationDataError(f"Invalid v2 metadata in {path}: {exc}") from exc
    if metadata.get("schema_version") != "2.0":
        raise HallucinationDataError(
            f"Unsupported Hallucination schema_version in {path}: "
            f"{metadata.get('schema_version')!r}"
        )
    if metadata.get("axis") != "hallucination":
        raise HallucinationDataError(f"V2 envelope {path} must use axis='hallucination'")
    phases = {expected_phase} if isinstance(expected_phase, str) else expected_phase
    if metadata.get("phase") not in phases:
        raise HallucinationDataError(
            f"V2 envelope {path} phase must be one of {sorted(phases)!r}, "
            f"got {metadata.get('phase')!r}"
        )


def generated_response_exclusion_reason(record: GeneratedResponseRecord) -> str | None:
    """Return one deterministic reason a generated row is detector-ineligible."""
    if record.error:
        return "generation_error"
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    response_metadata = (
        metadata.get("model_response_metadata")
        if isinstance(metadata.get("model_response_metadata"), dict)
        else {}
    )
    if metadata.get("parse_error") or response_metadata.get("parse_error"):
        return "parse_error"
    if record.response is None:
        return "null_response"
    if not isinstance(record.response, str):
        return "invalid_response_type"
    if not record.response.strip():
        return "blank_response"
    return None


def _parse_label(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError as exc:
            raise HallucinationDataError(f"Unable to parse label value {value!r}") from exc
    raise HallucinationDataError(f"Unable to parse label type {type(value).__name__}")


def _apply_max_samples(records: list[RecordT], max_samples: int | None) -> list[RecordT]:
    limit = resolve_max_samples(max_samples, len(records))
    return records[:limit]


def _ensure_unique_identities(identities: list[str], *, path: Path) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for identity in identities:
        if identity in seen:
            duplicates.add(identity)
        seen.add(identity)
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:5])
        raise HallucinationDataError(f"Duplicate stable identities in {path}: {preview}")


def _model_from_path(path: Path) -> str | None:
    match = re.match(r"results_(?P<model>.+?)_\d{8}_\d{6}\.json$", path.name)
    return match.group("model") if match else None


def _model_from_row(row: dict[str, Any], *, explicit_model: str | None, path: Path) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    model = explicit_model or metadata.get("model") or row.get("model") or row.get("Model") or _model_from_path(path)
    if not model:
        raise HallucinationDataError(
            f"Unable to determine model for generated response row in {path}; "
            "pass model=... or include metadata.model"
        )
    return str(model)


def load_native_prompt_response_json(
    path: str | Path,
    *,
    max_samples: int | None = None,
    prompt_key: str = "Prompt",
    response_key: str = "Response",
    row_id_key: str = "row_idx",
    parse_label: bool = True,
) -> list[PromptResponseRecord]:
    """Load native prompt/response rows using the configured stable identity key."""
    file_path = Path(path)
    rows = _require_list(_load_json(file_path), path=file_path)
    records: list[PromptResponseRecord] = []
    source_keys = {prompt_key, response_key, row_id_key}
    for index, raw in enumerate(rows):
        row = _require_mapping(raw, path=file_path, row=index)
        prompt = _require_text(row, prompt_key, path=file_path, index=index)
        response = _require_text(row, response_key, path=file_path, index=index)
        row_idx = _require_key(row, row_id_key, path=file_path, index=index)
        identity = str(row_idx)
        label = None
        if "Hallucination/Accuracy" in row:
            try:
                label = _parse_label(row.get("Hallucination/Accuracy"))
            except HallucinationDataError:
                if parse_label:
                    raise
        records.append(
            PromptResponseRecord(
                identity=identity,
                prompt=prompt,
                response=response,
                row_idx=row_idx,
                model=str(row["Model"]) if row.get("Model") is not None else None,
                label=label,
                additional_comments=row.get("Additional comments"),
                metadata={k: v for k, v in row.items() if k not in source_keys},
                source_row=dict(row),
                source_path=str(file_path),
            )
        )
    _ensure_unique_identities([record.identity for record in records], path=file_path)
    return _apply_max_samples(records, max_samples)


def load_legacy_generated_response_envelope(
    path: str | Path,
    *,
    model: str | None = None,
    max_samples: int | None = None,
) -> GeneratedResponseEnvelope:
    """Load legacy or v2 generated-response JSON without rewriting either schema."""
    file_path = Path(path)
    payload = _require_mapping(_load_json(file_path), path=file_path)
    if "results" not in payload:
        raise HallucinationDataError(
            f"Generated response envelope {file_path} must contain 'results' and 'statistics'"
        )
    results = _require_list(payload["results"], path=file_path, name="results")

    envelope_metadata: dict[str, Any] = {}
    schema_version: str | None = None
    is_v2 = False
    if "statistics" in payload:
        statistics = _require_mapping(payload["statistics"], path=file_path)
    elif "metadata" in payload and "summary" in payload:
        is_v2 = True
        envelope_metadata = _require_mapping(payload["metadata"], path=file_path)
        statistics = _require_mapping(payload["summary"], path=file_path)
        _validate_v2_envelope_metadata(
            envelope_metadata,
            path=file_path,
            expected_phase="response_generation",
        )
        schema_version = "2.0"
    else:
        raise HallucinationDataError(
            f"Generated response envelope {file_path} must contain 'results' and 'statistics'"
        )

    records: list[GeneratedResponseRecord] = []
    for ordinal, raw in enumerate(results):
        row = _require_mapping(raw, path=file_path, row=ordinal)
        prompt = _require_text(row, "prompt", path=file_path, index=ordinal)
        response = _require_key(row, "response", path=file_path, index=ordinal)
        row_model = _model_from_row(row, explicit_model=model, path=file_path)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if is_v2:
            source_identity = (
                row.get("sample_id")
                if row.get("sample_id") is not None
                else row.get("row_idx")
                if row.get("row_idx") is not None
                else metadata.get("source_identity")
                if metadata.get("source_identity") is not None
                else metadata.get("source_ordinal", ordinal)
            )
        else:
            source_identity = metadata.get("source_ordinal", ordinal)
        parsed_ordinal: int | None
        try:
            parsed_ordinal = int(source_identity)
        except (TypeError, ValueError):
            parsed_ordinal = None
        identity = f"{row_model}:{source_identity}"
        records.append(
            GeneratedResponseRecord(
                identity=identity,
                prompt=prompt,
                response=response,
                model=row_model,
                ordinal=parsed_ordinal,
                metadata=dict(metadata),
                error=row.get("error"),
                source_row=dict(row),
                source_path=str(file_path),
            )
        )
    _ensure_unique_identities([record.identity for record in records], path=file_path)
    records = _apply_max_samples(records, max_samples)
    return GeneratedResponseEnvelope(
        path=str(file_path),
        statistics=dict(statistics),
        responses=records,
        metadata=dict(envelope_metadata),
        schema_version=schema_version,
    )


def _detector_common_fields(
    row: dict[str, Any],
    *,
    path: Path,
    index: int,
) -> tuple[str | list[str] | None, str | None, list[dict[str, Any]]]:
    if row.get("detection_error"):
        return None, None, []
    merged_codes = normalize_merged_codes(_require_key(row, "merged_codes", path=path, index=index))
    rationale = row.get("rationale")
    if rationale is not None and not isinstance(rationale, str):
        raise HallucinationDataError(f"rationale must be a string or null in {path} row {index}")
    decisions = _require_key(row, "agent_decisions", path=path, index=index)
    if not isinstance(decisions, list):
        raise HallucinationDataError(f"agent_decisions must be a list in {path} row {index}")
    if not all(isinstance(decision, dict) for decision in decisions):
        raise HallucinationDataError(f"agent_decisions entries must be objects in {path} row {index}")
    return merged_codes, rationale, [dict(decision) for decision in decisions]


def _load_detector_payload(
    path: Path,
) -> tuple[list[Any], dict[str, Any], dict[str, Any], str | None]:
    payload = _load_json(path)
    if isinstance(payload, list):
        return payload, {}, {}, None
    if isinstance(payload, dict) and "results" in payload:
        if "metadata" not in payload or "summary" not in payload:
            raise HallucinationDataError(
                f"Detector v2 envelope {path} requires metadata, summary, and results"
            )
        rows = _require_list(payload["results"], path=path, name="results")
        metadata = _require_mapping(payload["metadata"], path=path)
        summary = _require_mapping(payload["summary"], path=path)
        _validate_v2_envelope_metadata(
            metadata,
            path=path,
            expected_phase={"native_validation", "generated_response_detection"},
        )
        return rows, metadata, summary, "2.0"
    raise HallucinationDataError(
        f"Expected top-level JSON in {path} to be a list or v2 envelope with results"
    )


def load_native_detector_output_list(
    path: str | Path,
    *,
    max_samples: int | None = None,
) -> list[DetectorRecord]:
    """Load native detector output rows with uppercase Prompt/Response and row_idx identity."""
    file_path = Path(path)
    rows, _, _, _ = _load_detector_payload(file_path)
    records: list[DetectorRecord] = []
    for index, raw in enumerate(rows):
        row = _require_mapping(raw, path=file_path, row=index)
        prompt_key = "prompt" if "prompt" in row else "Prompt"
        response_key = "response" if "response" in row else "Response"
        prompt = _require_text(row, prompt_key, path=file_path, index=index)
        response = _require_text(row, response_key, path=file_path, index=index)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        row_idx = (
            row.get("row_idx")
            if row.get("row_idx") is not None
            else metadata.get("source_identity")
        )
        if row_idx is None:
            raise HallucinationDataError(
                f"Missing native detector identity in {file_path} row {index}"
            )
        merged_codes, rationale, decisions = _detector_common_fields(row, path=file_path, index=index)
        label = None
        if "Hallucination/Accuracy" in row:
            try:
                label = _parse_label(row.get("Hallucination/Accuracy"))
            except HallucinationDataError:
                pass
        records.append(
            DetectorRecord(
                identity=str(row_idx),
                prompt=prompt,
                response=response,
                merged_codes=merged_codes,
                rationale=rationale,
                agent_decisions=decisions,
                execution_audit=(
                    dict(row["execution_audit"])
                    if isinstance(row.get("execution_audit"), dict)
                    else None
                ),
                row_idx=row_idx,
                model=str(row["Model"]) if row.get("Model") is not None else None,
                label=label,
                metadata={k: v for k, v in row.items() if k not in {"Prompt", "Response"}},
                error=row.get("detection_error", row.get("error")),
                source_row=dict(row),
                source_path=str(file_path),
                source_format="native",
            )
        )
    _ensure_unique_identities([record.identity for record in records], path=file_path)
    return _apply_max_samples(records, max_samples)


def load_generated_response_detector_list(
    path: str | Path,
    *,
    model: str | None = None,
    max_samples: int | None = None,
) -> list[DetectorRecord]:
    """Load generated-response detector rows with lowercase prompt/response."""
    file_path = Path(path)
    rows, _, _, _ = _load_detector_payload(file_path)
    records: list[DetectorRecord] = []
    for ordinal, raw in enumerate(rows):
        row = _require_mapping(raw, path=file_path, row=ordinal)
        prompt = _require_text(row, "prompt", path=file_path, index=ordinal)
        response = _require_text(row, "response", path=file_path, index=ordinal)
        row_model = _model_from_row(row, explicit_model=model, path=file_path)
        merged_codes, rationale, decisions = _detector_common_fields(row, path=file_path, index=ordinal)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        explicit_identity = metadata.get("source_identity")
        source_ordinal = metadata.get("source_ordinal")
        parsed_ordinal: int | None = None
        if source_ordinal is not None:
            try:
                parsed_ordinal = int(source_ordinal)
            except (TypeError, ValueError) as exc:
                raise HallucinationDataError(
                    f"metadata.source_ordinal must be an integer in {file_path} row {ordinal}"
                ) from exc
        if explicit_identity is not None:
            identity = str(explicit_identity)
        elif parsed_ordinal is not None:
            identity = f"{row_model}:{parsed_ordinal}"
        else:
            identity = f"{row_model}:artifact_row:{ordinal}"
        records.append(
            DetectorRecord(
                identity=identity,
                prompt=prompt,
                response=response,
                merged_codes=merged_codes,
                rationale=rationale,
                agent_decisions=decisions,
                execution_audit=(
                    dict(row["execution_audit"])
                    if isinstance(row.get("execution_audit"), dict)
                    else None
                ),
                model=row_model,
                ordinal=parsed_ordinal,
                metadata=dict(metadata),
                error=row.get("detection_error", row.get("error")),
                source_row=dict(row),
                source_path=str(file_path),
                source_format="generated",
            )
        )
    _ensure_unique_identities([record.identity for record in records], path=file_path)
    return _apply_max_samples(records, max_samples)


def load_detector_outputs(
    path: str | Path,
    *,
    source_format: Literal["auto", "native", "generated"] = "auto",
    model: str | None = None,
    max_samples: int | None = None,
) -> DetectorOutputList:
    """Load legacy or v2 detector rows, inferring native vs generated when requested."""
    file_path = Path(path)
    rows, metadata, summary, schema_version = _load_detector_payload(file_path)

    resolved_format: SourceFormat
    if source_format in {"native", "generated"}:
        resolved_format = source_format
    elif source_format == "auto":
        if rows:
            first = _require_mapping(rows[0], path=file_path, row=0)
            if "Prompt" in first and "Response" in first:
                resolved_format = "native"
            elif "prompt" in first and "response" in first:
                resolved_format = "generated"
            else:
                raise HallucinationDataError(
                    f"Cannot infer detector source format for {file_path}: "
                    "expected Prompt/Response or prompt/response keys"
                )
        else:
            phase = metadata.get("phase")
            resolved_format = "generated" if phase == "generated_response_detection" else "native"
    else:
        raise HallucinationDataError("source_format must be 'auto', 'native', or 'generated'")

    if resolved_format == "native":
        records = load_native_detector_output_list(file_path, max_samples=max_samples)
    else:
        records = load_generated_response_detector_list(
            file_path,
            model=model,
            max_samples=max_samples,
        )
    return DetectorOutputList(
        path=str(file_path),
        source_format=resolved_format,
        records=records,
        metadata=dict(metadata),
        summary=dict(summary),
        schema_version=schema_version,
    )


def load_prompt_responses(
    path: str | Path,
    *,
    source_format: Literal["auto", "native", "generated"] = "auto",
    model: str | None = None,
    max_samples: int | None = None,
    prompt_key: str = "Prompt",
    response_key: str = "Response",
    row_id_key: str = "row_idx",
    parse_label: bool = True,
) -> list[PromptResponseRecord] | GeneratedResponseEnvelope:
    """Load prompt/response inputs, inferring native lists vs generated envelopes."""
    file_path = Path(path)
    if source_format == "native":
        return load_native_prompt_response_json(
            file_path,
            max_samples=max_samples,
            prompt_key=prompt_key,
            response_key=response_key,
            row_id_key=row_id_key,
            parse_label=parse_label,
        )
    if source_format == "generated":
        return load_legacy_generated_response_envelope(file_path, model=model, max_samples=max_samples)
    if source_format != "auto":
        raise HallucinationDataError("source_format must be 'auto', 'native', or 'generated'")

    payload = _load_json(file_path)
    if isinstance(payload, dict) and "results" in payload:
        return load_legacy_generated_response_envelope(file_path, model=model, max_samples=max_samples)
    if isinstance(payload, list):
        return load_native_prompt_response_json(
            file_path,
            max_samples=max_samples,
            prompt_key=prompt_key,
            response_key=response_key,
            row_id_key=row_id_key,
            parse_label=parse_label,
        )
    raise HallucinationDataError(f"Cannot infer prompt/response format for {file_path}")


__all__ = [
    "DetectorOutputList",
    "DetectorRecord",
    "GeneratedResponseEnvelope",
    "GeneratedResponseRecord",
    "HallucinationDataError",
    "PromptResponseRecord",
    "SourceFormat",
    "dataset_sha256",
    "generated_response_exclusion_reason",
    "load_detector_outputs",
    "load_generated_response_detector_list",
    "load_legacy_generated_response_envelope",
    "load_native_detector_output_list",
    "load_native_prompt_response_json",
    "load_prompt_responses",
    "sha256_file",
    "sha256_json_value",
]
