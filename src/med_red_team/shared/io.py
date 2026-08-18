"""Non-destructive JSON and output-path mechanics."""

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
import json
import os
from pathlib import Path
import tempfile
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[3]
_IMMUTABLE_HALLUCINATION_ROOT = (_REPO_ROOT / "artifacts" / "hallucination").resolve()


def to_jsonable(value: Any) -> Any:
    """Recursively convert common framework values to JSON-safe objects."""
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_jsonable(to_dict())
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")


def _is_repo_relative_immutable_hallucination_path(path: Path) -> bool:
    normalized = Path(os.path.normpath(path.as_posix()))
    return normalized.parts[:2] == ("artifacts", "hallucination")


def _is_absolute_immutable_hallucination_path(path: Path) -> bool:
    try:
        lexical = Path(os.path.abspath(path))
        lexical.relative_to(_IMMUTABLE_HALLUCINATION_ROOT)
        return True
    except (OSError, ValueError):
        pass

    try:
        resolved_parent = path.parent.resolve(strict=False)
        resolved_parent.relative_to(_IMMUTABLE_HALLUCINATION_ROOT)
        return True
    except (OSError, ValueError):
        pass

    try:
        resolved = path.resolve(strict=False)
        resolved.relative_to(_IMMUTABLE_HALLUCINATION_ROOT)
        return True
    except (OSError, ValueError):
        return False


def _reject_immutable_hallucination_write(path: Path) -> None:
    if _is_repo_relative_immutable_hallucination_path(path) or _is_absolute_immutable_hallucination_path(path):
        raise ValueError(
            "Refusing to write under artifacts/hallucination; bundled release artifacts are read-only. "
            "Write runtime outputs elsewhere, such as logs/ or a temporary directory."
        )


def paths_alias(first: str | Path, second: str | Path) -> bool:
    """Return whether two paths name the same lexical, resolved, or existing file."""
    first_path = Path(first).expanduser()
    second_path = Path(second).expanduser()
    try:
        if first_path.resolve(strict=False) == second_path.resolve(strict=False):
            return True
    except OSError:
        if Path(os.path.abspath(first_path)) == Path(os.path.abspath(second_path)):
            return True
    if first_path.exists() and second_path.exists():
        try:
            return os.path.samefile(first_path, second_path)
        except OSError:
            return False
    return False


def require_distinct_paths(
    input_path: str | Path,
    output_path: str | Path,
    *,
    input_label: str = "Input",
    output_label: str = "Output",
) -> None:
    """Reject workflows that would replace their own input artifact."""
    if paths_alias(input_path, output_path):
        raise ValueError(
            f"{input_label} and {output_label} must refer to different files: "
            f"{input_path!s}"
        )


def validate_output_path(
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate an output path without opening or removing the final target."""
    path = Path(output_path)
    _reject_immutable_hallucination_write(path)
    if path.exists():
        if path.is_dir():
            raise IsADirectoryError(f"Output path is a directory: {path}")
        if path.is_symlink():
            raise ValueError(f"Refusing to overwrite symlink output path: {path}")
        if not overwrite:
            raise FileExistsError(
                f"Output file already exists: {path}. Pass overwrite=True to replace it."
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".write-test-", delete=True):
            pass
    except OSError as exc:
        raise OSError(f"Output directory is not writable: {path.parent}") from exc
    return path


def atomic_write_json(
    output_path: str | Path,
    data: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """Write JSON through a unique same-directory temporary file and replace."""
    path = Path(output_path)
    _reject_immutable_hallucination_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(to_jsonable(data), handle, indent=indent, ensure_ascii=ensure_ascii)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


_REQUIRED_METADATA_KEYS = {
    "schema_version",
    "axis",
    "phase",
    "config",
    "models",
    "dataset",
    "source",
    "is_partial",
}
_METADATA_OBJECT_KEYS = {"config", "models", "dataset", "source"}


def ensure_result_metadata(
    metadata: dict[str, Any] | None,
    *,
    axis: str,
    phase: str | None = None,
) -> dict[str, Any]:
    """Normalize common v2 metadata without interpreting axis-specific fields."""
    result = dict(metadata or {})
    result.setdefault("schema_version", "2.0")
    result.setdefault("axis", axis)
    if result["axis"] != axis:
        raise ValueError(
            f"Result metadata axis mismatch: expected {axis!r}, "
            f"got {result['axis']!r}"
        )

    resolved_phase = phase or result.get("phase") or result.get("mode")
    if not isinstance(resolved_phase, str) or not resolved_phase.strip():
        raise ValueError("Result metadata requires an explicit phase")
    result["phase"] = resolved_phase

    for key in _METADATA_OBJECT_KEYS:
        result.setdefault(key, {})
        if not isinstance(result[key], dict):
            raise TypeError(f"Result metadata {key!r} must be an object")

    result.setdefault("is_partial", False)
    if not isinstance(result["is_partial"], bool):
        raise TypeError("Result metadata 'is_partial' must be a boolean")
    return result


def validate_result_metadata(metadata: dict[str, Any]) -> None:
    """Validate the stable common metadata contract used by v2 artifacts."""
    missing = sorted(_REQUIRED_METADATA_KEYS.difference(metadata))
    if missing:
        raise ValueError(
            f"Result metadata is missing required keys: {', '.join(missing)}"
        )
    for key in _METADATA_OBJECT_KEYS:
        if not isinstance(metadata[key], dict):
            raise TypeError(f"Result metadata {key!r} must be an object")
    if not isinstance(metadata["is_partial"], bool):
        raise TypeError("Result metadata 'is_partial' must be a boolean")
    for key in ("schema_version", "axis", "phase"):
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise TypeError(f"Result metadata {key!r} must be a non-empty string")


def build_result_envelope(
    *,
    metadata: dict[str, Any],
    summary: Any,
    data: Any | None = None,
    data_key: str = "results",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the common envelope while preserving axis-local payload keys."""
    validate_result_metadata(metadata)
    if payload is not None and data is not None:
        raise ValueError("Pass either data or payload, not both")

    envelope = {
        "metadata": to_jsonable(metadata),
        "summary": to_jsonable(summary),
    }
    if payload is not None:
        envelope.update(to_jsonable(payload))
    else:
        envelope[data_key] = to_jsonable(data if data is not None else [])
    return envelope


def atomic_write_result_envelope(
    output_path: str | Path,
    *,
    metadata: dict[str, Any],
    summary: Any,
    data: Any | None = None,
    data_key: str = "results",
    payload: dict[str, Any] | None = None,
) -> None:
    """Build and atomically write a validated v2 result envelope."""
    atomic_write_json(
        output_path,
        build_result_envelope(
            metadata=metadata,
            summary=summary,
            data=data,
            data_key=data_key,
            payload=payload,
        ),
    )
