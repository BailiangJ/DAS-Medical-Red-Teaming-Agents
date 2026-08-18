#!/usr/bin/env python3
"""Validate packaged hallucination artifacts from the public manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from med_red_team.hallucination.artifact_validation import validate_hallucination_artifacts
from med_red_team.hallucination.config import ArtifactSummaryConfig, load_config
from med_red_team.shared.io import to_jsonable


DEFAULT_CONFIG_PATH = "configs/examples/hallucination/artifact_summary.py"


def _json_dumps(value: Any) -> str:
    return json.dumps(to_jsonable(value), indent=2, ensure_ascii=False, sort_keys=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate hallucination artifact manifest, hashes, counts, JSON, and public-safety scans.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.hallucination.validate_artifacts
  python -m scripts.hallucination.validate_artifacts --artifact-root artifacts/hallucination --json
  python -m scripts.hallucination.validate_artifacts --manifest artifacts/hallucination/manifest.json
        """,
    )
    parser.add_argument("--config", help=f"ArtifactSummaryConfig .py/.json path (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--artifact-root", help="Artifact root directory containing manifest/files")
    parser.add_argument("--manifest", help="Manifest path or artifact-root-relative manifest name")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args(argv)


def _manifest_root_and_name(manifest: str | None, artifact_root: str | None) -> tuple[str | None, str | None]:
    if manifest is None:
        return artifact_root, None

    manifest_path = Path(manifest)
    if artifact_root is None:
        if manifest_path.parent == Path("") or manifest_path.parent == Path("."):
            return None, manifest_path.as_posix()
        return manifest_path.parent.as_posix(), manifest_path.name

    root_path = Path(artifact_root)
    if manifest_path.is_absolute():
        try:
            manifest_name = manifest_path.resolve().relative_to(root_path.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("--manifest must be inside --artifact-root when both are supplied") from exc
        return root_path.as_posix(), manifest_name
    return root_path.as_posix(), manifest_path.as_posix()


def _load_config(args: argparse.Namespace) -> ArtifactSummaryConfig:
    if args.config:
        config = load_config(args.config, ArtifactSummaryConfig)
    elif Path(DEFAULT_CONFIG_PATH).is_file():
        config = load_config(DEFAULT_CONFIG_PATH, ArtifactSummaryConfig)
    else:
        config = ArtifactSummaryConfig()
    data = config.to_dict()

    root_override, manifest_name = _manifest_root_and_name(args.manifest, args.artifact_root)
    if root_override is not None:
        data["artifact_root"] = root_override
    if manifest_name is not None:
        data["manifest_name"] = manifest_name
    data["validate_hashes"] = True
    data["validate_json"] = True
    data["validate_counts"] = True
    data["scan_public_safety"] = True
    data["raise_on_error"] = False
    return ArtifactSummaryConfig.from_dict(data)


def _print_human(report: Any) -> None:
    print("Hallucination artifact validation")
    print(f"  Artifact root: {report.artifact_root}")
    print(f"  Manifest: {report.manifest_path}")
    print(f"  OK: {report.ok}")
    print(f"  Checks: {report.checks}")
    print(f"  Totals: {report.totals}")
    if report.errors:
        print("  Errors:")
        for issue in report.errors:
            print(f"    [{issue.severity}] {issue.path}: {issue.message}")
    if report.warnings:
        print("  Warnings:")
        for issue in report.warnings:
            print(f"    [{issue.severity}] {issue.path}: {issue.message}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = _load_config(args)
        report = validate_hallucination_artifacts(config, raise_on_error=False)
        if args.json:
            print(_json_dumps(report.to_dict()))
        else:
            _print_human(report)
        return 0 if report.ok else 1
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        if args.json:
            print(_json_dumps({"error": str(exc)}))
        else:
            print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
