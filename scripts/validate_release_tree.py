"""Validate that a checkout contains only releasable source-tree files.

The provider-free check operates on paths, file names, and bounded-memory text
streams. The Hallucination bundle is allowlisted
from its manifest; its hashes, contents, and physical scope are validated
separately by ``scripts.hallucination.validate_artifacts``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Iterable, Sequence


DEFAULT_MANIFEST = Path("artifacts/hallucination/manifest.json")

# These are generated during local runs and must never be part of a release
# tree.  The scan checks directory names, so an empty generated directory is
# rejected as well.
RUNTIME_DIRECTORIES = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".cache",
        "logs",
        "outputs",
        "results",
        "checkpoints",
        "checkpoint",
        "build",
        "dist",
        # Reproducibility policy: these names conventionally hold local,
        # private, or ephemeral run state even when they are otherwise empty.
        "private",
        "internal",
        "tmp",
        "runs",
    }
)

TEXT_FILE_SUFFIXES = frozenset(
    {
        ".cfg",
        ".conf",
        ".css",
        ".env",
        ".html",
        ".htm",
        ".ini",
        ".js",
        ".json",
        ".jsonl",
        ".md",
        ".py",
        ".rst",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsv",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)
TEXT_LIKENESS_SAMPLE_SIZE = 8 * 1024
TEXT_SCAN_CHUNK_SIZE = 64 * 1024
TEXT_SCAN_OVERLAP = 4096

BINARY_SUFFIXES = frozenset(
    {
        ".7z",
        ".avif",
        ".bmp",
        ".class",
        ".dll",
        ".dylib",
        ".exe",
        ".gif",
        ".gz",
        ".ico",
        ".jpeg",
        ".jpg",
        ".mp3",
        ".mp4",
        ".pdf",
        ".png",
        ".so",
        ".tar",
        ".tif",
        ".tiff",
        ".webp",
        ".woff",
        ".zip",
    }
)

WEIGHT_SUFFIXES = frozenset(
    {
        ".bin",
        ".ckpt",
        ".gguf",
        ".h5",
        ".joblib",
        ".onnx",
        ".pkl",
        ".pickle",
        ".pt",
        ".pth",
        ".safetensors",
    }
)

PRIVATE_DATA_SUFFIXES = frozenset(
    {
        ".csv",
        ".db",
        ".feather",
        ".jsonl",
        ".parquet",
        ".sqlite",
        ".sqlite3",
        ".tsv",
        ".xls",
        ".xlsx",
    }
)

RUNTIME_FILE_NAMES = frozenset(
    {
        "result.json",
        "results.json",
        "output.json",
        "outputs.json",
    }
)

PRIVATE_PATH_PATTERNS = (
    re.compile(
        r"(?<![A-Za-z0-9])/(?:home|Users)/[A-Za-z0-9._-]+"
        r"(?:/|(?=$|[\s\"'<>]))"
    ),
    re.compile(r"(?i)(?<![A-Za-z0-9])(?:[A-Z]:\\Users\\[A-Za-z0-9._-]+\\)"),
    re.compile(r"(?<![A-Za-z0-9])/(?:mnt/data|private/var)/"),
)

CREDENTIAL_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{20,}(?![0-9A-Za-z_-])"),
    re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}(?![A-Za-z0-9])"),
    re.compile(r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}(?![0-9A-Z])"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
CREDENTIAL_FILE_PATTERN = re.compile(
    r"(?:^|[._-])(?:credential|credentials|secret|secrets|token|tokens|"
    r"api[-_]?key|api[-_]?keys|private[-_]?key|private[-_]?keys)(?:[._-]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ReleaseViolation:
    """One release-tree policy violation."""

    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "message": self.message}


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _symlink_component(path: Path, root: Path) -> Path | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None

    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return current
    return None


def _safe_manifest_paths(root: Path, manifest_path: Path) -> set[str]:
    """Return the manifest and its declared non-symlink physical files, if present."""
    if _symlink_component(manifest_path, root) is not None or not manifest_path.is_file():
        return set()

    relative_manifest = _relative_path(manifest_path, root)
    manifest_parent = Path(relative_manifest).parent
    allowed = {relative_manifest}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        # The dedicated artifact validator reports malformed manifests.  Keep
        # this scan conservative when the manifest cannot be read.
        return allowed

    for entry in payload.get("files", []) if isinstance(payload, dict) else []:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            continue
        declared = Path(entry["path"])
        if declared.is_absolute() or ".." in declared.parts:
            continue
        relative_declared = manifest_parent / declared
        candidate = root / relative_declared
        if _symlink_component(candidate, root) is not None:
            continue
        try:
            candidate.resolve(strict=False).relative_to(root)
        except (OSError, ValueError):
            continue
        allowed.add(relative_declared.as_posix())
    return allowed


def _filesystem_paths(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if ".git" in path.relative_to(root).parts:
            continue
        yield path


def _tracked_paths(root: Path) -> Iterable[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    for raw_path in result.stdout.split(b"\0"):
        if raw_path:
            yield root / raw_path.decode("utf-8")


def _is_text_like(path: Path) -> bool:
    """Return whether ``path`` is safe to inspect as bounded text chunks.

    Known text suffixes are scanned regardless of size.  For unknown suffixes,
    inspect only a small prefix to avoid treating common binary assets as text;
    the full file is still consumed only through bounded chunks.
    """
    suffix = path.suffix.lower()
    if suffix in TEXT_FILE_SUFFIXES:
        return True
    if suffix in BINARY_SUFFIXES or suffix in WEIGHT_SUFFIXES:
        return False
    try:
        with path.open("rb") as handle:
            sample = handle.read(TEXT_LIKENESS_SAMPLE_SIZE)
    except OSError:
        return False
    return b"\x00" not in sample


def _contains_private_text(path: Path) -> str | None:
    """Scan text-like files without loading an unbounded file into memory."""
    if not path.is_file() or not _is_text_like(path):
        return None

    carry = ""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            while True:
                chunk = handle.read(TEXT_SCAN_CHUNK_SIZE)
                if not chunk:
                    break
                window = carry + chunk
                for pattern in PRIVATE_PATH_PATTERNS:
                    if pattern.search(window):
                        return "contains a private absolute path"
                for pattern in CREDENTIAL_PATTERNS:
                    if pattern.search(window):
                        return "contains a credential-like value"
                carry = window[-TEXT_SCAN_OVERLAP:]
    except OSError:
        return None
    return None


def scan_release_tree(
    root: str | Path = ".",
    *,
    manifest: str | Path = DEFAULT_MANIFEST,
    tracked_only: bool = False,
) -> list[ReleaseViolation]:
    """Return release-policy violations for ``root``.

    ``tracked_only`` is useful in CI before build/install steps create ignored
    files.  The default scans the physical tree and is useful as a local guard.
    Approved Hallucination files are read from the manifest allowlist before
    applying the generic output/secret policies.
    """
    root_path = Path(root).resolve()
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = root_path / manifest_path
    manifest_path = Path(os.path.abspath(manifest_path))
    try:
        manifest_path.relative_to(root_path)
    except ValueError as exc:
        raise ValueError("--manifest must be inside --root") from exc

    allowed_artifacts = _safe_manifest_paths(root_path, manifest_path)
    paths = _tracked_paths(root_path) if tracked_only else _filesystem_paths(root_path)
    violations: list[ReleaseViolation] = []
    reported_symlinks: set[str] = set()

    for path in paths:
        try:
            relative = _relative_path(path, root_path)
        except ValueError:
            continue

        symlink = _symlink_component(path, root_path)
        if symlink is not None:
            symlink_relative = _relative_path(symlink, root_path)
            if symlink_relative not in reported_symlinks:
                violations.append(
                    ReleaseViolation(
                        symlink_relative,
                        "symlinks are not releasable",
                    )
                )
                reported_symlinks.add(symlink_relative)
            continue

        # Directory entries that contain only manifest-declared files are also part of
        # the allowlisted artifact tree, even though directories are not
        # represented in manifest.files.
        if relative in allowed_artifacts or any(
            declared.startswith(relative + "/") for declared in allowed_artifacts
        ):
            continue

        parts = Path(relative).parts
        name = path.name.lower()
        suffix = path.suffix.lower()

        if parts and parts[0] == "artifacts":
            violations.append(
                ReleaseViolation(relative, "artifact is not declared by an artifact manifest")
            )
            continue

        generated_dir = next(
            (
                part
                for part in parts
                if part.lower() in RUNTIME_DIRECTORIES
                or part.lower().endswith(".egg-info")
            ),
            None,
        )
        if generated_dir is not None:
            violations.append(
                ReleaseViolation(relative, f"generated/runtime directory is not releasable: {generated_dir}")
            )
            continue

        if suffix in WEIGHT_SUFFIXES:
            violations.append(ReleaseViolation(relative, "model weights or serialized model data are not releasable"))
            continue
        if suffix in PRIVATE_DATA_SUFFIXES:
            violations.append(ReleaseViolation(relative, "private/local dataset or database file is not releasable"))
            continue
        if name in RUNTIME_FILE_NAMES or suffix in {".inprogress", ".tmp", ".log"}:
            violations.append(ReleaseViolation(relative, "runtime output/checkpoint file is not releasable"))
            continue
        if "checkpoint" in path.stem.lower() and suffix not in {".py", ".pyi"}:
            violations.append(ReleaseViolation(relative, "checkpoint output is not releasable"))
            continue
        if (
            name in {
                ".env",
                ".env.local",
                ".npmrc",
                ".pypirc",
                "credentials.json",
                "service-account.json",
            }
            or suffix in {".key", ".pem"}
            or CREDENTIAL_FILE_PATTERN.search(path.stem) is not None
        ):
            violations.append(ReleaseViolation(relative, "credential/config secret file is not releasable"))
            continue

        text_violation = _contains_private_text(path)
        if text_violation:
            violations.append(ReleaseViolation(relative, text_violation))

    return violations


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan a source tree for runtime outputs, private data, weights, and secrets."
    )
    parser.add_argument("--root", default=".", help="Source-tree root (default: current directory)")
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST.as_posix(),
        help="Approved Hallucination artifact manifest path",
    )
    parser.add_argument(
        "--tracked-only",
        action="store_true",
        help="Scan git-tracked paths instead of every physical path",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        violations = scan_release_tree(
            args.root,
            manifest=args.manifest,
            tracked_only=args.tracked_only,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        else:
            print(f"[ERROR] {exc}")
        return 1

    payload = {
        "ok": not violations,
        "violations": [violation.to_dict() for violation in violations],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif violations:
        print("Release-tree validation failed:")
        for violation in violations:
            print(f"  {violation.path}: {violation.message}")
    else:
        print("Release-tree validation passed")
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
