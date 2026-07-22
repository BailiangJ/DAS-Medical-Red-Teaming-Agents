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
