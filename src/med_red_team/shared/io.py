"""Non-destructive JSON and output-path mechanics."""

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
import json
import os
from pathlib import Path
import tempfile
from typing import Any


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


def validate_output_path(
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate an output path without opening or removing the final target."""
    path = Path(output_path)
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


def ensure_result_metadata(
    metadata: dict[str, Any] | None,
    *,
    axis: str,
) -> dict[str, Any]:
    """Add the small common v2 metadata envelope without replacing axis fields."""
    result = dict(metadata or {})
    result.setdefault("schema_version", "2.0")
    result.setdefault("axis", axis)
    result.setdefault("phase", result.get("mode", "unknown"))
    result.setdefault("config", {})
    result.setdefault(
        "models",
        {
            key: result[key]
            for key in (
                "target_model",
                "testee_model",
                "grader_model",
                "generator_model",
                "orchestrator_model",
                "tools_model",
            )
            if key in result
        },
    )
    result.setdefault(
        "dataset",
        result.get("dataset_info", result.get("data_file", {})),
    )
    result.setdefault("source", {})
    result.setdefault("is_partial", False)
    return result


def build_result_envelope(
    *,
    metadata: dict[str, Any],
    summary: Any,
    data: Any,
    data_key: str = "results",
) -> dict[str, Any]:
    """Build the minimal common envelope while preserving axis-local payloads."""
    required = {"schema_version", "axis", "phase", "config", "models", "dataset"}
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"Result metadata is missing required keys: {', '.join(missing)}")
    return {
        "metadata": to_jsonable(metadata),
        "summary": to_jsonable(summary),
        data_key: to_jsonable(data),
    }
