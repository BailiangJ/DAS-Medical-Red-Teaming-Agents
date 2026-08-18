#!/usr/bin/env python3
"""Offline summaries for packaged hallucination manuscript artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

from med_red_team.hallucination.artifact_validation import validate_hallucination_artifacts
from med_red_team.hallucination.config import ArtifactSummaryConfig, load_config
from med_red_team.hallucination.data import (
    DetectorRecord,
    load_generated_response_detector_list,
    load_native_detector_output_list,
)
from med_red_team.hallucination.metrics import (
    UncertainPolicy,
    binary_classification_report,
    chi_square_test,
    cohen_kappa,
    merged_code_to_binary,
    prediction_counts,
    rogan_gladen_from_counts,
    wilson_interval,
)
from med_red_team.hallucination.schemas import (
    SubAgentDecision,
    merge_agent_classifications,
    normalize_merged_codes,
)
from med_red_team.shared.io import read_json, to_jsonable


DEFAULT_CONFIG_PATH = "configs/examples/hallucination/artifact_summary.py"
_UNSET = object()

ROOT_CODE_NAMES = {
    "1": "medical_fact",
    "2": "citation",
    "3": "reasoning",
    "4": "context",
    "5": "safety",
    "6": "instruction",
    "7": "other_hallucination_or_bias",
}


def _json_dumps(value: Any) -> str:
    return json.dumps(to_jsonable(value), indent=2, ensure_ascii=False, sort_keys=True)


def _parse_max_samples(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"none", "all", "null"}:
        return None
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max samples must be an integer, 'all', or 'none'") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("max samples must be None/all, 0, or a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize hallucination artifacts offline without providers or detectors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.hallucination.summarize_artifacts
  python -m scripts.hallucination.summarize_artifacts --config configs/paper/hallucination/artifact_summary.py --json
  python -m scripts.hallucination.summarize_artifacts --include-chi-square --include-rogan-gladen
        """,
    )
    parser.add_argument("--config", help=f"ArtifactSummaryConfig .py/.json path (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--artifact-root", help="Override config.artifact_root")
    parser.add_argument("--manifest", help="Override manifest path or artifact-root-relative manifest name")
    parser.add_argument(
        "--max-samples",
        type=_parse_max_samples,
        default=_UNSET,
        help="Optional row cap for quick offline smoke checks; use all/none to clear a preset cap",
    )
    parser.add_argument(
        "--uncertain-policy",
        choices=["negative", "positive", "drop"],
        default="positive",
        help="How to binarize detector code 0.5 (default: positive)",
    )
    parser.add_argument("--include-chi-square", action="store_true", help="Optionally run scipy chi-square across generated model rates")
    parser.add_argument("--include-rogan-gladen", action="store_true", help="Optionally estimate corrected generated rates using primary native sensitivity/specificity")
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
    root = Path(artifact_root)
    if manifest_path.is_absolute():
        try:
            manifest_name = manifest_path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("--manifest must be inside --artifact-root when both are supplied") from exc
        return root.as_posix(), manifest_name
    return root.as_posix(), manifest_path.as_posix()


def _load_summary_config(args: argparse.Namespace) -> ArtifactSummaryConfig:
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
    if args.max_samples is not _UNSET:
        data["max_samples"] = args.max_samples
    return ArtifactSummaryConfig.from_dict(data)


def _rate(successes: int, total: int) -> dict[str, Any]:
    low, high = wilson_interval(successes, total) if total else (0.0, 0.0)
    return {
        "successes": successes,
        "total": total,
        "rate": successes / total if total else 0.0,
        "wilson_95": {"low": low, "high": high},
    }


def _report_with_intervals(records: Sequence[DetectorRecord], *, uncertain_policy: UncertainPolicy) -> dict[str, Any]:
    y_true = [record.label for record in records]
    y_pred = [record.merged_codes for record in records]
    report = binary_classification_report(y_true, y_pred, uncertain_policy=uncertain_policy)
    counts = report["counts"]
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]
    total = counts["total"]
    positives = tp + fp
    actual_positive = tp + fn
    actual_negative = tn + fp
    report["intervals"] = {
        "accuracy": _rate(tp + tn, total),
        "precision": _rate(tp, positives),
        "recall_sensitivity": _rate(tp, actual_positive),
        "specificity": _rate(tn, actual_negative),
        "predicted_positive_rate": _rate(positives, total),
    }
    return report


def _concat(*groups: Sequence[DetectorRecord]) -> list[DetectorRecord]:
    combined: list[DetectorRecord] = []
    for group in groups:
        combined.extend(group)
    return combined


def _records_by_row_idx(records: Sequence[DetectorRecord]) -> dict[str, DetectorRecord]:
    return {str(record.row_idx): record for record in records if record.row_idx is not None}


def _kappa_between(
    left: Sequence[DetectorRecord],
    right: Sequence[DetectorRecord],
    *,
    uncertain_policy: UncertainPolicy,
) -> dict[str, Any]:
    left_by_id = _records_by_row_idx(left)
    right_by_id = _records_by_row_idx(right)
    overlap = sorted(set(left_by_id).intersection(right_by_id), key=lambda item: (len(item), item))
    left_binary = [merged_code_to_binary(left_by_id[row_idx].merged_codes, uncertain_policy=uncertain_policy) for row_idx in overlap]
    right_codes = [right_by_id[row_idx].merged_codes for row_idx in overlap]
    return {
        "left_count": len(left_by_id),
        "right_count": len(right_by_id),
        "overlap_count": len(overlap),
        "left_only_count": len(set(left_by_id).difference(right_by_id)),
        "right_only_count": len(set(right_by_id).difference(left_by_id)),
        "kappa": cohen_kappa(left_binary, right_codes, uncertain_policy=uncertain_policy),
    }


def _kappa_between_dataset_groups(
    groups: Sequence[tuple[str, Sequence[DetectorRecord], Sequence[DetectorRecord]]],
    *,
    uncertain_policy: UncertainPolicy,
) -> dict[str, Any]:
    """Combine detector agreement without colliding row_idx across datasets."""
    left_binary: list[int | None] = []
    right_codes: list[Any] = []
    overlaps: dict[str, int] = {}
    left_only: dict[str, int] = {}
    right_only: dict[str, int] = {}
    left_counts: dict[str, int] = {}
    right_counts: dict[str, int] = {}
    for namespace, left, right in groups:
        left_by_id = _records_by_row_idx(left)
        right_by_id = _records_by_row_idx(right)
        overlap = sorted(set(left_by_id).intersection(right_by_id), key=lambda item: (len(item), item))
        overlaps[namespace] = len(overlap)
        left_counts[namespace] = len(left_by_id)
        right_counts[namespace] = len(right_by_id)
        left_only[namespace] = len(set(left_by_id).difference(right_by_id))
        right_only[namespace] = len(set(right_by_id).difference(left_by_id))
        left_binary.extend(
            merged_code_to_binary(left_by_id[row_idx].merged_codes, uncertain_policy=uncertain_policy)
            for row_idx in overlap
        )
        right_codes.extend(right_by_id[row_idx].merged_codes for row_idx in overlap)
    return {
        "left_count": sum(left_counts.values()),
        "right_count": sum(right_counts.values()),
        "overlap_count": sum(overlaps.values()),
        "left_count_by_dataset": left_counts,
        "right_count_by_dataset": right_counts,
        "overlap_by_dataset": overlaps,
        "left_only_by_dataset": left_only,
        "right_only_by_dataset": right_only,
        "kappa": cohen_kappa(left_binary, right_codes, uncertain_policy=uncertain_policy),
    }


def _legacy_consistency_counts(records: Iterable[DetectorRecord]) -> dict[str, int]:
    """Report consistency between stored specialist decisions and merged codes."""
    comparable = 0
    mismatched = 0
    unavailable = 0
    for record in records:
        try:
            decisions = [
                SubAgentDecision.model_validate(decision)
                for decision in record.agent_decisions
            ]
            expected = normalize_merged_codes(
                merge_agent_classifications(decisions)
            )
            actual = normalize_merged_codes(record.merged_codes)
        except (TypeError, ValueError):
            unavailable += 1
            continue
        comparable += 1
        if expected != actual:
            mismatched += 1
    return {
        "comparable_count": comparable,
        "mismatch_count": mismatched,
        "unavailable_count": unavailable,
    }


def _root_code_counts(records: Iterable[DetectorRecord]) -> dict[str, int]:
    counts = {code: 0 for code in ROOT_CODE_NAMES}
    for record in records:
        merged = normalize_merged_codes(record.merged_codes)
        if isinstance(merged, list):
            for code in set(merged):
                counts[str(code)] += 1
    return counts


def _uncertain_count(records: Iterable[DetectorRecord]) -> int:
    total = 0
    for record in records:
        if normalize_merged_codes(record.merged_codes) == "0.5":
            total += 1
    return total


def _declared_artifact_path(
    root: Path,
    manifest: dict[str, Any],
    relative_path: str,
) -> Path | None:
    entries = _manifest_entries(manifest, "")
    if any(str(entry.get("path")) == relative_path for entry in entries):
        return root / relative_path
    return None


def _manifest_entries(manifest: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    entries = [entry for entry in files if isinstance(entry, dict) and str(entry.get("path", "")).startswith(prefix)]
    return sorted(entries, key=lambda entry: str(entry.get("path", "")))


def _native_summary(
    root: Path,
    manifest: dict[str, Any],
    *,
    max_samples: int | None,
    uncertain_policy: UncertainPolicy,
) -> dict[str, Any]:
    def load_declared(relative_path: str) -> list[DetectorRecord]:
        path = _declared_artifact_path(root, manifest, relative_path)
        return (
            load_native_detector_output_list(path, max_samples=max_samples)
            if path is not None
            else []
        )

    openai_stanford = load_declared(
        "native_detector/openai_gpt4o_o3/stanford_positive_131/detailed_outputs.json"
    )
    openai_healthbench = load_declared(
        "native_detector/openai_gpt4o_o3/healthbench_negative_manuscript_129/detailed_outputs.json"
    )
    claude_stanford = load_declared(
        "native_detector/cross_vendor_claude/stanford_positive_131/detailed_outputs.json"
    )
    claude_healthbench = load_declared(
        "native_detector/cross_vendor_claude/healthbench_negative_131/detailed_outputs.json"
    )

    openai_combined = _concat(openai_stanford, openai_healthbench)
    openai_healthbench_ids = {str(record.row_idx) for record in openai_healthbench}
    claude_healthbench_manuscript = [
        record for record in claude_healthbench if str(record.row_idx) in openai_healthbench_ids
    ]
    claude_manuscript_combined = _concat(claude_stanford, claude_healthbench_manuscript)
    claude_combined = _concat(claude_stanford, claude_healthbench)
    primary_report = _report_with_intervals(openai_combined, uncertain_policy=uncertain_policy)
    claude_manuscript_report = _report_with_intervals(
        claude_manuscript_combined,
        uncertain_policy=uncertain_policy,
    )
    claude_report = _report_with_intervals(claude_combined, uncertain_policy=uncertain_policy)
    return {
        "legacy_consistency_diagnostics": {
            "primary_stanford": _legacy_consistency_counts(openai_stanford),
            "primary_healthbench": _legacy_consistency_counts(openai_healthbench),
            "comparative_stanford": _legacy_consistency_counts(claude_stanford),
            "comparative_healthbench": _legacy_consistency_counts(claude_healthbench),
            "metric_source": "artifact_merged_codes",
        },
        "primary_openai_gpt4o_o3": {
            "stanford_positive_131": _report_with_intervals(openai_stanford, uncertain_policy=uncertain_policy),
            "healthbench_negative_manuscript_129": _report_with_intervals(openai_healthbench, uncertain_policy=uncertain_policy),
            "combined_validation": primary_report,
        },
        "claude_comparative": {
            "stanford_positive_131": _report_with_intervals(claude_stanford, uncertain_policy=uncertain_policy),
            "healthbench_negative_131": _report_with_intervals(claude_healthbench, uncertain_policy=uncertain_policy),
            "combined_validation": claude_report,
            "manuscript_260_validation": claude_manuscript_report,
            "kappa_against_labels": {
                "stanford_positive_131": cohen_kappa(
                    [record.label for record in claude_stanford],
                    [record.merged_codes for record in claude_stanford],
                    uncertain_policy=uncertain_policy,
                ),
                "healthbench_negative_131": cohen_kappa(
                    [record.label for record in claude_healthbench],
                    [record.merged_codes for record in claude_healthbench],
                    uncertain_policy=uncertain_policy,
                ),
                "combined_validation": cohen_kappa(
                    [record.label for record in claude_combined],
                    [record.merged_codes for record in claude_combined],
                    uncertain_policy=uncertain_policy,
                ),
                "manuscript_260_validation": cohen_kappa(
                    [record.label for record in claude_manuscript_combined],
                    [record.merged_codes for record in claude_manuscript_combined],
                    uncertain_policy=uncertain_policy,
                ),
            },
            "kappa_against_primary_openai": {
                "stanford_overlap": _kappa_between(openai_stanford, claude_stanford, uncertain_policy=uncertain_policy),
                "healthbench_overlap": _kappa_between(openai_healthbench, claude_healthbench, uncertain_policy=uncertain_policy),
                "combined_overlap": _kappa_between_dataset_groups(
                    [
                        ("stanford", openai_stanford, claude_stanford),
                        ("healthbench", openai_healthbench, claude_healthbench),
                    ],
                    uncertain_policy=uncertain_policy,
                ),
            },
        },
    }


def _generated_response_counts(root: Path, manifest: dict[str, Any], *, max_samples: int | None) -> dict[str, Any]:
    files = _manifest_entries(manifest, "generated_responses/")
    by_file: dict[str, Any] = {}
    total_loaded = 0
    total_declared = 0
    total_usable = 0
    total_unavailable = 0
    total_null = 0
    total_blank = 0
    total_errors = 0
    for entry in files:
        rel_path = str(entry["path"])
        payload = read_json(root / rel_path)
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError(f"Generated response artifact must contain a results list: {rel_path}")
        results = [row for row in payload["results"] if isinstance(row, dict)]
        selected = results if max_samples is None else results[:max_samples]
        statistics = payload.get("statistics") if isinstance(payload.get("statistics"), dict) else {}
        declared = entry.get("row_count")
        models = set()
        null_response_count = 0
        blank_response_count = 0
        error_count = 0
        unavailable_identities: set[int] = set()
        for row_index, row in enumerate(selected):
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            model = metadata.get("model") or row.get("model") or row.get("Model")
            if model is not None:
                models.add(str(model))
            response = row.get("response")
            if response is None:
                null_response_count += 1
                unavailable_identities.add(row_index)
            elif not isinstance(response, str) or not response.strip():
                blank_response_count += 1
                unavailable_identities.add(row_index)
            if row.get("error"):
                error_count += 1
                unavailable_identities.add(row_index)
        unavailable_count = len(unavailable_identities)
        usable_count = len(selected) - unavailable_count
        total_loaded += len(selected)
        total_declared += int(declared or 0)
        total_usable += usable_count
        total_unavailable += unavailable_count
        total_null += null_response_count
        total_blank += blank_response_count
        total_errors += error_count
        by_file[rel_path] = {
            "loaded_count": len(selected),
            "manifest_row_count": declared,
            "statistics_total_prompts": statistics.get("total_prompts"),
            "statistics_total_responses": statistics.get("total_responses"),
            "models_observed": sorted(models),
            "usable_response_count": usable_count,
            "unavailable_response_count": unavailable_count,
            "null_response_count": null_response_count,
            "blank_response_count": blank_response_count,
            "error_count": error_count,
        }
    return {
        "file_count": len(files),
        "loaded_rows_total": total_loaded,
        "manifest_rows_total": total_declared,
        "usable_response_rows_total": total_usable,
        "unavailable_response_rows_total": total_unavailable,
        "null_response_rows_total": total_null,
        "blank_response_rows_total": total_blank,
        "error_rows_total": total_errors,
        "by_file": by_file,
    }


def _generated_detection_summary(
    root: Path,
    manifest: dict[str, Any],
    *,
    max_samples: int | None,
    uncertain_policy: UncertainPolicy,
) -> dict[str, Any]:
    files = _manifest_entries(manifest, "generated_response_detection/openai_gpt4o_o3/")
    denominators = manifest.get("generated_response_detection_denominators", {})
    if not isinstance(denominators, dict):
        denominators = {}

    by_model: dict[str, Any] = {}
    aggregate_root_counts = {code: 0 for code in ROOT_CODE_NAMES}
    aggregate_positive = 0
    aggregate_negative = 0
    aggregate_raw = 0
    aggregate_evaluated = 0
    aggregate_dropped = 0
    chi_square_rows: list[list[int]] = []

    for entry in files:
        rel_path = str(entry["path"])
        model_dir = Path(rel_path).parts[-2]
        records = load_generated_response_detector_list(root / rel_path, max_samples=max_samples)
        manifest_denominator = denominators.get(model_dir, {}) if isinstance(denominators.get(model_dir), dict) else {}
        counts = prediction_counts(
            (record.merged_codes for record in records),
            uncertain_policy=uncertain_policy,
        )
        denominator = counts["evaluated"]
        positive = counts["positive"]
        negative = counts["negative"]
        root_counts = _root_code_counts(records)
        for code, count in root_counts.items():
            aggregate_root_counts[code] += count
        aggregate_positive += positive
        aggregate_negative += negative
        aggregate_raw += counts["raw"]
        aggregate_evaluated += counts["evaluated"]
        aggregate_dropped += counts["dropped"]
        chi_square_rows.append([positive, negative])
        by_model[model_dir] = {
            "path": rel_path,
            "detected_response_models_observed": entry.get("detected_response_models_observed", sorted({record.model for record in records})),
            "loaded_count": len(records),
            "manifest_row_count": entry.get("row_count"),
            "manifest_denominator": manifest_denominator,
            "selected_raw_count": counts["raw"],
            "selected_evaluated_count": counts["evaluated"],
            "dropped_count": counts["dropped"],
            "positive_detection": _rate(positive, denominator),
            "negative_count": negative,
            "uncertain_count": _uncertain_count(records),
            "expected_source_count": manifest_denominator.get(
                "expected_source_response_count"
            ),
            "artifact_available_count": manifest_denominator.get(
                "actual_evaluation_denominator"
            ),
            "missing_source_count": manifest_denominator.get(
                "missing_from_detection_count"
            ),
            "coverage_rate": (
                manifest_denominator["actual_evaluation_denominator"]
                / manifest_denominator["expected_source_response_count"]
                if manifest_denominator.get("expected_source_response_count")
                else 0.0
            ),
            "root_code_counts": root_counts,
            "legacy_consistency_diagnostics": _legacy_consistency_counts(records),
            "root_code_rates": {
                code: {
                    "name": ROOT_CODE_NAMES[code],
                    **_rate(count, denominator),
                }
                for code, count in root_counts.items()
            },
        }

    return {
        "file_count": len(files),
        "loaded_rows_total": aggregate_raw,
        "raw_rows_total": aggregate_raw,
        "evaluated_rows_total": aggregate_evaluated,
        "dropped_rows_total": aggregate_dropped,
        "positive_detection_total": aggregate_positive,
        "negative_detection_total": aggregate_negative,
        "overall_positive_detection": _rate(
            aggregate_positive,
            aggregate_evaluated,
        ),
        "aggregate_root_code_counts": aggregate_root_counts,
        "aggregate_root_code_rates": {
            code: {"name": ROOT_CODE_NAMES[code], **_rate(count, aggregate_evaluated)}
            for code, count in aggregate_root_counts.items()
        },
        "by_model": by_model,
        "chi_square_table_model_by_positive_negative": chi_square_rows,
    }


def _optional_analysis(
    summary: dict[str, Any],
    *,
    include_chi_square: bool,
    include_rogan_gladen: bool,
) -> dict[str, Any]:
    analysis: dict[str, Any] = {}
    generated = summary["generated_response_detection"]
    table = generated["chi_square_table_model_by_positive_negative"]
    if include_chi_square:
        try:
            analysis["chi_square_generated_model_rates"] = chi_square_test(table)
        except ImportError as exc:
            analysis["chi_square_generated_model_rates"] = {"available": False, "reason": str(exc)}
        except ValueError as exc:
            analysis["chi_square_generated_model_rates"] = {"available": False, "reason": str(exc)}

    if include_rogan_gladen:
        primary_counts = summary["native_detector_metrics"]["primary_openai_gpt4o_o3"]["combined_validation"]["counts"]
        tp = primary_counts["tp"]
        fp = primary_counts["fp"]
        tn = primary_counts["tn"]
        fn = primary_counts["fn"]
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        corrected_by_model: dict[str, Any] = {}
        for model, data in generated["by_model"].items():
            rate = data["positive_detection"]
            try:
                corrected_by_model[model] = rogan_gladen_from_counts(
                    int(rate["successes"]),
                    int(rate["total"]),
                    sensitivity,
                    specificity,
                )
            except ValueError as exc:
                corrected_by_model[model] = {"available": False, "reason": str(exc)}
        analysis["rogan_gladen_generated_rates_using_primary_native"] = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "by_model": corrected_by_model,
        }
    return analysis


def summarize(config: ArtifactSummaryConfig, *, uncertain_policy: UncertainPolicy, include_chi_square: bool, include_rogan_gladen: bool) -> dict[str, Any]:
    validation = validate_hallucination_artifacts(
        config,
        raise_on_error=True,
    )
    root = Path(config.artifact_root)
    manifest = read_json(config.manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a JSON object: {config.manifest_path}")
    max_samples = config.max_samples
    native_metrics = _native_summary(
        root,
        manifest,
        max_samples=max_samples,
        uncertain_policy=uncertain_policy,
    )
    summary = {
        "artifact_root": config.artifact_root,
        "manifest_path": config.manifest_path.as_posix(),
        "max_samples_applied": max_samples,
        "uncertain_policy": uncertain_policy,
        "artifact_validation": {
            "ok": validation.ok,
            "checks": validation.checks,
            "totals": validation.totals,
            "warning_count": len(validation.warnings),
        },
        "manifest_totals": manifest.get("totals", {}),
        "native_detector_metrics": native_metrics,
        "generated_response_counts": _generated_response_counts(root, manifest, max_samples=max_samples),
        "generated_response_detection": _generated_detection_summary(
            root,
            manifest,
            max_samples=max_samples,
            uncertain_policy=uncertain_policy,
        ),
        "known_caveats": manifest.get("global_known_caveats", []),
    }
    summary["optional_analysis"] = _optional_analysis(
        summary,
        include_chi_square=include_chi_square,
        include_rogan_gladen=include_rogan_gladen,
    )
    return summary


def _print_human(summary: dict[str, Any]) -> None:
    print("Hallucination artifact summary")
    print(f"  Artifact root: {summary['artifact_root']}")
    print(f"  Manifest: {summary['manifest_path']}")
    print(f"  Max samples applied: {summary['max_samples_applied']}")
    primary = summary["native_detector_metrics"]["primary_openai_gpt4o_o3"]["combined_validation"]
    primary_metrics = primary["metrics"]
    print("  Primary OpenAI native combined metrics:")
    print(f"    accuracy={primary_metrics['accuracy']:.4f} precision={primary_metrics['precision']:.4f} recall={primary_metrics['recall']:.4f} f1={primary_metrics['f1']:.4f}")
    claude = summary["native_detector_metrics"]["claude_comparative"]["combined_validation"]
    claude_metrics = claude["metrics"]
    claude_kappa = summary["native_detector_metrics"]["claude_comparative"]["kappa_against_primary_openai"]["combined_overlap"]
    print("  Claude comparative combined metrics:")
    print(f"    accuracy={claude_metrics['accuracy']:.4f} precision={claude_metrics['precision']:.4f} recall={claude_metrics['recall']:.4f} f1={claude_metrics['f1']:.4f}")
    print(f"    kappa_vs_primary_openai={claude_kappa['kappa']:.4f} on {claude_kappa['overlap_count']} overlapping row_idx")
    generated_counts = summary["generated_response_counts"]
    print("  Generated responses:")
    print(f"    files={generated_counts['file_count']} loaded_rows={generated_counts['loaded_rows_total']} manifest_rows={generated_counts['manifest_rows_total']}")
    generated_detection = summary["generated_response_detection"]
    overall = generated_detection["overall_positive_detection"]
    print("  Generated-response detections:")
    print(f"    files={generated_detection['file_count']} loaded_rows={generated_detection['loaded_rows_total']} positive_rate={overall['rate']:.4f} ({overall['successes']}/{overall['total']})")
    if summary["optional_analysis"]:
        print("  Optional analysis:")
        for key, value in summary["optional_analysis"].items():
            print(f"    {key}: {value}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = _load_summary_config(args)
        summary = summarize(
            config,
            uncertain_policy=args.uncertain_policy,
            include_chi_square=args.include_chi_square,
            include_rogan_gladen=args.include_rogan_gladen,
        )
        if args.json:
            print(_json_dumps(summary))
        else:
            _print_human(summary)
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        if args.json:
            print(_json_dumps({"error": str(exc)}))
        else:
            print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
