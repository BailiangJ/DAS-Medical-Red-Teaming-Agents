"""Offline validation for packaged hallucination manuscript artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from med_red_team.shared.io import read_json, to_jsonable

from .config import ArtifactSummaryConfig
from .data import sha256_file


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    path: str
    message: str


@dataclass
class ArtifactValidationReport:
    artifact_root: str
    manifest_path: str
    ok: bool
    checks: dict[str, Any] = field(default_factory=dict)
    totals: dict[str, Any] = field(default_factory=dict)
    validated_files: dict[str, str] = field(default_factory=dict)
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


class ArtifactValidationError(ValueError):
    """Raised when artifact validation fails in strict mode."""

    def __init__(self, report: ArtifactValidationReport):
        self.report = report
        messages = [f"{issue.path}: {issue.message}" for issue in report.errors[:10]]
        suffix = "" if len(report.errors) <= 10 else f"; plus {len(report.errors) - 10} more"
        super().__init__("Hallucination artifact validation failed: " + "; ".join(messages) + suffix)


_PRIVATE_PATH_RE = re.compile(r"(?:^|[\s\"'])((?:/home|/Users)/[^\s\"']+|[A-Za-z]:\\\\Users\\\\[^\s\"']+)")
_CREDENTIAL_RE = re.compile(
    r"sk-[A-Za-z0-9]{20,}|"
    r"AKIA[0-9A-Z]{16}|"
    r"AIza[0-9A-Za-z_-]{20,}|"
    r"gh[pousr]_[A-Za-z0-9_]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{20,}|"
    r"(?i:(?:api[_-]?key|secret|token|password)\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{16,})"
)
_GOOGLE_ADK_RE = re.compile(r"\bgoogle\.adk\b|\bgoogle_adk\b|\bgoogle-adk\b", re.IGNORECASE)


def _add_issue(
    report: ArtifactValidationReport,
    severity: str,
    path: str | Path,
    message: str,
) -> None:
    issue = ValidationIssue(severity=severity, path=str(path), message=message)
    if severity == "error":
        report.errors.append(issue)
    else:
        report.warnings.append(issue)


def _safe_relative_path(path_value: str, *, report: ArtifactValidationReport) -> Path | None:
    rel = Path(path_value)
    if rel.is_absolute() or ".." in rel.parts:
        _add_issue(report, "error", path_value, "manifest file path must be relative and stay under artifact_root")
        return None
    return rel


def _symlink_component(root: Path, rel: Path) -> Path | None:
    current = root
    if current.is_symlink():
        return current
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            return current
    return None


def _validated_artifact_path(
    root: Path,
    rel: Path,
    *,
    report: ArtifactValidationReport,
) -> Path | None:
    candidate = root / rel
    symlink = _symlink_component(root, rel)
    if symlink is not None:
        _add_issue(
            report,
            "error",
            rel,
            f"artifact path contains a symlink component: {symlink}",
        )
        return None

    try:
        resolved_root = root.resolve(strict=False)
        candidate.resolve(strict=False).relative_to(resolved_root)
    except (OSError, ValueError):
        _add_issue(
            report,
            "error",
            rel,
            "artifact path resolves outside artifact_root",
        )
        return None
    return candidate


def _row_model_value(row: Any, *, fallback: str | None = None) -> str | None:
    if not isinstance(row, dict):
        return None
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    value = metadata.get("model") or row.get("model") or row.get("Model") or fallback
    return str(value) if value is not None else None


def _model_from_generated_filename(path: str) -> str | None:
    match = re.match(r"results_(.+?)_\d{8}_\d{6}\.json$", Path(path).name)
    return match.group(1) if match else None


def _json_row_count(value: Any) -> int | None:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict) and isinstance(value.get("results"), list):
        return len(value["results"])
    return None


def _parse_json_file(path: Path, *, report: ArtifactValidationReport) -> Any | None:
    try:
        return read_json(path)
    except json.JSONDecodeError as exc:
        _add_issue(report, "error", path, f"JSON parse failed: {exc}")
    except OSError as exc:
        _add_issue(report, "error", path, f"Unable to read JSON file: {exc}")
    return None


def _scan_public_safety(path: Path, *, report: ArtifactValidationReport) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        _add_issue(report, "error", path, f"Unable to scan file: {exc}")
        return

    private_match = _PRIVATE_PATH_RE.search(text)
    if private_match:
        _add_issue(report, "error", path, f"private absolute path-like string found: {private_match.group(1)}")
    credential_match = _CREDENTIAL_RE.search(text)
    if credential_match:
        _add_issue(report, "error", path, "credential-like token found")
    adk_match = _GOOGLE_ADK_RE.search(text)
    if adk_match:
        _add_issue(report, "error", path, "reference excluded by artifact policy")


def _physical_file_allowlist(root: Path, *, report: ArtifactValidationReport) -> set[str]:
    paths: set[str] = set()
    try:
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if path.is_symlink():
                _add_issue(
                    report,
                    "error",
                    relative,
                    "symlinks are not allowed in the immutable artifact tree",
                )
                continue
            if path.is_file():
                paths.add(relative.as_posix())
    except OSError as exc:
        _add_issue(report, "error", root, f"Unable to enumerate artifact files: {exc}")
    return paths


def _compare_value(
    report: ArtifactValidationReport,
    *,
    path: str,
    observed: Any,
    expected: Any,
    name: str,
) -> None:
    if observed != expected:
        _add_issue(report, "error", path, f"{name} expected {expected!r}, observed {observed!r}")


def validate_hallucination_artifacts(
    config: ArtifactSummaryConfig | str | Path | None = None,
    *,
    raise_on_error: bool | None = None,
) -> ArtifactValidationReport:
    """Validate manifest integrity, counts, hashes, JSON parse, and public-safety scans."""
    if config is None:
        cfg = ArtifactSummaryConfig()
    elif isinstance(config, ArtifactSummaryConfig):
        cfg = config
    else:
        cfg = ArtifactSummaryConfig(artifact_root=str(config))

    root = Path(cfg.artifact_root)
    manifest_path = cfg.manifest_path
    report = ArtifactValidationReport(
        artifact_root=str(root),
        manifest_path=str(manifest_path),
        ok=False,
    )

    manifest_rel = _safe_relative_path(cfg.manifest_name, report=report)
    validated_manifest_path = (
        _validated_artifact_path(root, manifest_rel, report=report)
        if manifest_rel is not None
        else None
    )
    if validated_manifest_path is None:
        if raise_on_error if raise_on_error is not None else cfg.raise_on_error:
            raise ArtifactValidationError(report)
        return report
    manifest_path = validated_manifest_path

    manifest = _parse_json_file(manifest_path, report=report)
    if not isinstance(manifest, dict):
        _add_issue(report, "error", manifest_path, "manifest must be a JSON object")
        report.ok = False
        if raise_on_error if raise_on_error is not None else cfg.raise_on_error:
            raise ArtifactValidationError(report)
        return report

    files = manifest.get("files")
    if not isinstance(files, list):
        _add_issue(report, "error", manifest_path, "manifest.files must be a list")
        files = []

    if cfg.scan_public_safety:
        _scan_public_safety(manifest_path, report=report)

    row_counts: dict[str, int | None] = {}
    parsed_payloads: dict[str, Any] = {}
    declared_paths: set[str] = set()
    validated_files: dict[str, Path] = {}
    hash_checked = 0
    json_checked = 0
    for entry_index, entry in enumerate(files):
        if not isinstance(entry, dict):
            _add_issue(report, "error", manifest_path, f"manifest.files[{entry_index}] must be an object")
            continue
        path_value = entry.get("path")
        if not isinstance(path_value, str):
            _add_issue(report, "error", manifest_path, f"manifest.files[{entry_index}].path must be a string")
            continue
        rel = _safe_relative_path(path_value, report=report)
        if rel is None:
            continue
        rel_path = rel.as_posix()
        if rel_path in declared_paths:
            _add_issue(report, "error", manifest_path, f"manifest.files repeats path {rel_path!r}")
            continue
        declared_paths.add(rel_path)
        full_path = _validated_artifact_path(root, rel, report=report)
        if full_path is None:
            continue
        if not full_path.is_file():
            _add_issue(report, "error", rel, "manifest file is missing")
            continue
        validated_files[rel_path] = full_path

        if cfg.validate_hashes:
            expected_hash = entry.get("sha256")
            expected_size = entry.get("size_bytes")
            if isinstance(expected_hash, str):
                actual_hash = sha256_file(full_path)
                hash_checked += 1
                if actual_hash != expected_hash:
                    _add_issue(report, "error", rel, f"sha256 mismatch: expected {expected_hash}, observed {actual_hash}")
            else:
                _add_issue(report, "error", rel, "manifest entry missing sha256")
            if isinstance(expected_size, int):
                actual_size = full_path.stat().st_size
                if actual_size != expected_size:
                    _add_issue(report, "error", rel, f"size mismatch: expected {expected_size}, observed {actual_size}")

        parsed_json = None
        if cfg.validate_json and full_path.suffix == ".json":
            parsed_json = _parse_json_file(full_path, report=report)
            parsed_payloads[rel_path] = parsed_json
            json_checked += 1
            actual_count = _json_row_count(parsed_json)
            row_counts[rel_path] = actual_count
            expected_count = entry.get("row_count")
            if expected_count is not None and actual_count != expected_count:
                _add_issue(report, "error", rel, f"row_count expected {expected_count}, observed {actual_count}")
        else:
            row_counts[rel_path] = entry.get("row_count") if isinstance(entry.get("row_count"), int) else None

        if cfg.scan_public_safety:
            _scan_public_safety(full_path, report=report)

    manifest_rel_path = Path(cfg.manifest_name).as_posix()
    actual_paths = _physical_file_allowlist(root, report=report)
    allowed_paths = set(declared_paths)
    allowed_paths.add(manifest_rel_path)
    for unexpected in sorted(actual_paths.difference(allowed_paths)):
        _add_issue(
            report,
            "error",
            unexpected,
            "physical artifact file is not declared in manifest.files",
        )

    report.validated_files = {
        path: str(full_path) for path, full_path in validated_files.items()
    }
    report.checks.update(
        {
            "manifest_loaded": True,
            "manifest_file_entries": len(files),
            "hashes_checked": hash_checked,
            "json_files_checked": json_checked,
            "public_safety_scan": cfg.scan_public_safety,
            "manifest_self_scanned": cfg.scan_public_safety,
            "physical_allowlist_checked": True,
            "physical_file_count": len(actual_paths),
            "allowed_file_count": len(allowed_paths),
        }
    )

    if cfg.validate_counts:
        totals = manifest.get("totals", {}) if isinstance(manifest.get("totals"), dict) else {}
        report.totals.update(dict(totals))
        generated_entries = [
            entry
            for entry in files
            if isinstance(entry, dict)
            and str(entry.get("path", "")).startswith("generated_responses/")
        ]
        detection_entries = [
            entry
            for entry in files
            if isinstance(entry, dict)
            and str(entry.get("path", "")).startswith("generated_response_detection/")
        ]
        declared_generated_files = totals.get("generated_response_file_count")
        declared_detection_files = totals.get(
            "generated_response_detection_file_count"
        )
        _compare_value(
            report,
            path="manifest.files",
            observed=len(generated_entries),
            expected=declared_generated_files,
            name="generated response manifest entry count",
        )
        _compare_value(
            report,
            path="manifest.files",
            observed=len(detection_entries),
            expected=declared_detection_files,
            name="generated response detection manifest entry count",
        )

        denominators = manifest.get("generated_response_detection_denominators")
        if not isinstance(denominators, dict):
            _add_issue(
                report,
                "error",
                "manifest.generated_response_detection_denominators",
                "must be an object",
            )
            denominators = {}
        response_counts_by_model: dict[str, int | None] = {}
        for entry in generated_entries:
            rel_path = str(entry.get("path", ""))
            if not cfg.validate_json:
                declared_model = entry.get("model_or_detector_backend")
                if isinstance(declared_model, str):
                    if declared_model in response_counts_by_model:
                        _add_issue(report, "error", rel_path, f"duplicate generated-response source model {declared_model!r}")
                    else:
                        response_counts_by_model[declared_model] = entry.get("row_count")
                continue
            payload = parsed_payloads.get(rel_path)
            rows = (
                payload.get("results")
                if isinstance(payload, dict) and isinstance(payload.get("results"), list)
                else []
            )
            fallback_model = _model_from_generated_filename(rel_path)
            resolved_models = [
                _row_model_value(row, fallback=fallback_model) for row in rows
            ]
            if any(model is None for model in resolved_models):
                _add_issue(report, "error", rel_path, "every generated-response row must resolve to a model")
                continue
            observed_models = {str(model) for model in resolved_models if model is not None}
            if len(observed_models) != 1:
                _add_issue(report, "error", rel_path, "generated-response file must contain exactly one observed model")
                continue
            observed_model = next(iter(observed_models))
            if observed_model in response_counts_by_model:
                _add_issue(report, "error", rel_path, f"duplicate generated-response source model {observed_model!r}")
                continue
            response_counts_by_model[observed_model] = entry.get("row_count")
        seen_model_dirs: set[str] = set()
        for entry in detection_entries:
            rel_path = str(entry.get("path", ""))
            model_dir = entry.get("detected_response_model_dir")
            if not isinstance(model_dir, str) or not model_dir:
                _add_issue(report, "error", rel_path, "missing detected_response_model_dir")
                continue
            if model_dir in seen_model_dirs:
                _add_issue(report, "error", rel_path, f"duplicate detected_response_model_dir {model_dir!r}")
                continue
            seen_model_dirs.add(model_dir)
            actual = entry.get("actual_evaluation_denominator")
            expected_source = entry.get("expected_source_response_count")
            missing = entry.get("missing_from_detection_count")
            if not all(isinstance(value, int) and not isinstance(value, bool) for value in (actual, expected_source, missing)):
                _add_issue(report, "error", rel_path, "detection denominator fields must be integers")
                continue
            if any(value < 0 for value in (actual, expected_source, missing)):
                _add_issue(report, "error", rel_path, "detection denominator fields must be non-negative")
                continue
            declared_models = entry.get("detected_response_models_observed")
            payload = parsed_payloads.get(rel_path)
            rows = (
                payload
                if isinstance(payload, list)
                else payload.get("results", [])
                if isinstance(payload, dict) and isinstance(payload.get("results"), list)
                else []
            )
            resolved_artifact_models = [
                _row_model_value(row) for row in rows
            ]
            if actual > 0 and any(
                model is None for model in resolved_artifact_models
            ):
                _add_issue(
                    report,
                    "error",
                    rel_path,
                    "every generated detection row must resolve to a source model",
                )
            artifact_models = {
                str(model)
                for model in resolved_artifact_models
                if model is not None
            }
            if not isinstance(declared_models, list) or len(declared_models) != 1 or not isinstance(declared_models[0], str):
                _add_issue(report, "error", rel_path, "detected_response_models_observed must contain exactly one model")
            else:
                if (
                    cfg.validate_json
                    and actual > 0
                    and artifact_models != {declared_models[0]}
                ):
                    _add_issue(report, "error", rel_path, "detected response models in artifact do not match manifest declaration")
                source_count = response_counts_by_model.get(declared_models[0])
                if source_count is None:
                    _add_issue(report, "error", rel_path, "detected response model has no generated-response source entry")
                elif expected_source != source_count:
                    _add_issue(report, "error", rel_path, "expected source count does not match generated-response source entry")
            if actual + missing != expected_source:
                _add_issue(report, "error", rel_path, "actual denominator plus missing count must equal expected source count")
            _compare_value(
                report,
                path=rel_path,
                observed=entry.get("row_count"),
                expected=actual,
                name="row_count versus actual_evaluation_denominator",
            )
            top_level = denominators.get(model_dir)
            if top_level != {
                "actual_evaluation_denominator": actual,
                "expected_source_response_count": expected_source,
                "missing_from_detection_count": missing,
            }:
                _add_issue(
                    report,
                    "error",
                    f"manifest.generated_response_detection_denominators.{model_dir}",
                    "does not match the corresponding manifest file entry",
                )
        if set(denominators) != seen_model_dirs:
            _add_issue(
                report,
                "error",
                "manifest.generated_response_detection_denominators",
                "keys must match detected response model directories",
            )

        declared_dataset_rows = {
            "datasets/stanford_redteaming_131_positive_cases.json": totals.get(
                "stanford_current_dataset_rows"
            ),
            "datasets/healthbench_131_negative_cases.json": totals.get(
                "healthbench_current_dataset_rows"
            ),
        }
        for rel_path, declared in declared_dataset_rows.items():
            if rel_path in declared_paths:
                _compare_value(
                    report,
                    path=rel_path,
                    observed=row_counts.get(rel_path),
                    expected=declared,
                    name="row_count versus manifest total",
                )

        generated_total = sum(
            count or 0
            for path, count in row_counts.items()
            if path.startswith("generated_responses/")
        )
        generated_detection_total = sum(
            count or 0
            for path, count in row_counts.items()
            if path.startswith("generated_response_detection/")
        )
        _compare_value(
            report,
            path="computed.generated_responses",
            observed=generated_total,
            expected=totals.get("generated_response_rows_total"),
            name="generated response row total",
        )
        _compare_value(
            report,
            path="computed.generated_response_detection",
            observed=generated_detection_total,
            expected=totals.get("generated_response_detection_rows_total"),
            name="generated-response detection row total",
        )
        report.totals["computed_generated_response_rows_total"] = generated_total
        report.totals["computed_generated_response_detection_rows_total"] = generated_detection_total

    report.ok = not report.errors
    should_raise = cfg.raise_on_error if raise_on_error is None else raise_on_error
    if should_raise and report.errors:
        raise ArtifactValidationError(report)
    return report


__all__ = [
    "ArtifactValidationError",
    "ArtifactValidationReport",
    "ValidationIssue",
    "validate_hallucination_artifacts",
]
