from __future__ import annotations

import ast
import re
from pathlib import Path

from med_red_team.hallucination.artifact_validation import validate_hallucination_artifacts
from med_red_team.hallucination.config import ArtifactSummaryConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "hallucination"
PUBLIC_CODE_ROOTS = [
    REPO_ROOT / "src" / "med_red_team" / "hallucination",
    REPO_ROOT / "scripts" / "hallucination",
    REPO_ROOT / "configs" / "examples" / "hallucination",
    REPO_ROOT / "configs" / "paper" / "hallucination",
]

_GOOGLE_ADK_IMPORTS = {"google.adk", "google_adk"}
_GOOGLE_ADK_PACKAGE_RE = re.compile(r"\bgoogle[._-]adk\b", re.IGNORECASE)
_PRIVATE_PATH_RE = re.compile(r"(?:^|[\s\"'])((?:/home|/Users)/[^\s\"']+|[A-Za-z]:\\\\Users\\\\[^\s\"']+)")
_CREDENTIAL_RE = re.compile(
    r"sk-[A-Za-z0-9]{20,}|"
    r"AKIA[0-9A-Z]{16}|"
    r"AIza[0-9A-Za-z_-]{20,}|"
    r"gh[pousr]_[A-Za-z0-9_]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{20,}|"
    r"(?i:(?:api[_-]?key|secret|token|password)\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{16,})"
)


def _python_files() -> list[Path]:
    files: list[Path] = []
    for root in PUBLIC_CODE_ROOTS:
        files.extend(sorted(root.rglob("*.py")))
    return files


def test_hallucination_public_code_does_not_import_google_adk_packages():
    offenders: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported = alias.name
                    if imported in _GOOGLE_ADK_IMPORTS or imported.startswith("google.adk."):
                        offenders.append(f"{path}: import {imported}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in _GOOGLE_ADK_IMPORTS or module.startswith("google.adk."):
                    offenders.append(f"{path}: from {module} import ...")

    assert offenders == []


def test_hallucination_public_code_has_no_hardcoded_private_paths_or_credentials():
    offenders: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            value = node.value
            # Scanner regex literals are fixtures, not matched credential values.
            if "(?:/home|/Users)" in value or "api[_-]?key" in value or "google\\.adk" in value:
                continue
            private = _PRIVATE_PATH_RE.search(value)
            credential = _CREDENTIAL_RE.search(value)
            if private:
                offenders.append(f"{path}: private path {private.group(1)!r}")
            if credential:
                offenders.append(f"{path}: credential-like token")

    assert offenders == []


def test_artifact_metadata_public_safety_scan_and_exclusions():
    report = validate_hallucination_artifacts(
        ArtifactSummaryConfig(artifact_root=str(ARTIFACT_ROOT), scan_public_safety=True),
        raise_on_error=False,
    )
    assert report.ok, [issue.message for issue in report.errors]

    metadata_files = [ARTIFACT_ROOT / "manifest.json", ARTIFACT_ROOT / "README.md"]
    metadata_files.extend((ARTIFACT_ROOT / "datasets").glob("*_manifest.json"))
    metadata_files.append(ARTIFACT_ROOT / "native_detector" / "cross_vendor_claude" / "provenance.json")

    offenders: list[str] = []
    for path in metadata_files:
        text = path.read_text(encoding="utf-8")
        if _GOOGLE_ADK_PACKAGE_RE.search(text):
            offenders.append(f"{path}: actual Google ADK package token")
        if _PRIVATE_PATH_RE.search(text):
            offenders.append(f"{path}: private absolute path")
        if _CREDENTIAL_RE.search(text):
            offenders.append(f"{path}: credential-like token")

    assert offenders == []
