"""Checkpoint file primitives; payload semantics remain axis-owned."""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar

from med_red_team.shared.io import atomic_write_json, read_json


ItemT = TypeVar("ItemT")
KeyT = TypeVar("KeyT")


def checkpoint_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_name(f"{path.name}.inprogress")


def write_checkpoint(output_path: str | Path, payload: Any) -> Path:
    path = checkpoint_path(output_path)
    atomic_write_json(path, payload)
    return path


def load_checkpoint(output_path: str | Path) -> Any | None:
    path = checkpoint_path(output_path)
    return read_json(path) if path.exists() else None


def remove_checkpoint(output_path: str | Path) -> None:
    path = checkpoint_path(output_path)
    if path.exists():
        path.unlink()


def retire_foreign_checkpoint(
    resume_path: str | Path | None,
    authoritative_path: str | Path,
    *,
    enabled: bool = False,
) -> bool:
    """Remove a foreign resume checkpoint only when explicitly requested."""
    if not enabled or resume_path is None:
        return False
    resume = Path(resume_path)
    authoritative = Path(authoritative_path)
    if (
        resume.expanduser().resolve() == authoritative.expanduser().resolve()
        or not resume.name.endswith(".inprogress")
        or not resume.exists()
        or not authoritative.exists()
    ):
        return False
    resume.unlink()
    return True


def merge_unique(
    existing: Iterable[ItemT],
    current: Iterable[ItemT],
    *,
    key: Callable[[ItemT], KeyT],
) -> list[ItemT]:
    """Merge items by an axis-provided identity, preferring current values."""
    merged: dict[KeyT, ItemT] = {}
    for item in existing:
        merged[key(item)] = item
    for item in current:
        merged[key(item)] = item
    return list(merged.values())


def require_complete_artifact(
    metadata: dict[str, Any],
    *,
    label: str = "Input artifact",
) -> None:
    """Reject a partial artifact used as completed upstream scientific input."""
    if metadata.get("is_partial") is not False:
        raise ValueError(f"{label} must be complete (is_partial=false)")


def validate_resume_metadata(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    keys: Iterable[str] = ("schema_version", "axis", "phase"),
) -> None:
    mismatches = [
        name for name in keys
        if expected.get(name) != actual.get(name)
    ]
    if mismatches:
        details = ", ".join(
            f"{name}: expected {expected.get(name)!r}, got {actual.get(name)!r}"
            for name in mismatches
        )
        raise ValueError(f"Checkpoint is incompatible with this run ({details})")
