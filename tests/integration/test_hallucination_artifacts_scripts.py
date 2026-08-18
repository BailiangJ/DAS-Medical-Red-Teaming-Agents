from __future__ import annotations

import json
from pathlib import Path

from med_red_team.hallucination.artifact_validation import validate_hallucination_artifacts
from med_red_team.hallucination.config import ArtifactSummaryConfig
from med_red_team.shared.io import read_json
from scripts.hallucination import summarize_artifacts, validate_artifacts


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "hallucination"


def test_packaged_hallucination_artifact_validation_hashes_counts_and_exclusions():
    report = validate_hallucination_artifacts(
        ArtifactSummaryConfig(artifact_root=str(ARTIFACT_ROOT)),
        raise_on_error=False,
    )

    assert report.ok, [issue.message for issue in report.errors]
    assert report.checks["manifest_loaded"] is True
    manifest = read_json(ARTIFACT_ROOT / "manifest.json")
    json_entry_count = sum(1 for entry in manifest["files"] if entry["path"].endswith(".json"))
    assert report.checks["hashes_checked"] == report.checks["manifest_file_entries"]
    assert report.checks["json_files_checked"] == json_entry_count
    assert report.checks["public_safety_scan"] is True
    assert report.checks["manifest_self_scanned"] is True
    assert report.checks["physical_allowlist_checked"] is True
    assert report.checks["physical_file_count"] == report.checks["allowed_file_count"]
    assert report.totals["generated_response_rows_total"] == 1965
    assert report.totals["computed_generated_response_rows_total"] == 1965
    assert report.totals["generated_response_detection_rows_total"] == 1918
    assert report.totals["computed_generated_response_detection_rows_total"] == 1918
    assert report.totals["stanford_current_dataset_rows"] == 131
    assert report.totals["healthbench_current_dataset_rows"] == 131


def test_count_validation_can_run_without_json_parsing():
    report = validate_hallucination_artifacts(
        ArtifactSummaryConfig(
            artifact_root=str(ARTIFACT_ROOT),
            validate_hashes=False,
            validate_json=False,
            validate_counts=True,
            scan_public_safety=False,
        ),
        raise_on_error=False,
    )

    assert report.ok, [issue.message for issue in report.errors]


def test_manifest_file_counts_match_packaged_artifact_scope():
    manifest = read_json(ARTIFACT_ROOT / "manifest.json")
    files = manifest["files"]

    generated_response_entries = [entry for entry in files if entry["path"].startswith("generated_responses/")]
    generated_detection_entries = [entry for entry in files if entry["path"].startswith("generated_response_detection/")]
    native_entries = [entry for entry in files if entry["path"].startswith("native_detector/")]

    assert manifest["totals"]["generated_response_file_count"] == 15
    assert len(generated_response_entries) == 15
    assert sum(entry["row_count"] for entry in generated_response_entries) == 1965
    assert manifest["totals"]["generated_response_detection_file_count"] == 15
    assert len(generated_detection_entries) == 15
    assert sum(entry["row_count"] for entry in generated_detection_entries) == 1918
    assert {entry["row_count"] for entry in native_entries if entry["row_count"] is not None} >= {129, 131}

    all_manifest_paths = "\n".join(json.dumps(entry, sort_keys=True) for entry in files)
    assert "/home/" not in all_manifest_paths
    assert "/Users/" not in all_manifest_paths
    assert "C:\\Users\\" not in all_manifest_paths
    assert "google.adk" not in all_manifest_paths.lower()
    assert "google_adk" not in all_manifest_paths.lower()
    assert "google-adk" not in all_manifest_paths.lower()


def test_artifact_validation_rejects_negative_denominator_fields(tmp_path):
    root = tmp_path / "artifacts"
    source_path = root / "generated_responses" / "source.json"
    detection_path = root / "generated_response_detection" / "model" / "detailed_outputs.json"
    detection_path.parent.mkdir(parents=True)
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        json.dumps({"results": [{"prompt": "p", "response": "r"}], "statistics": {}}),
        encoding="utf-8",
    )
    detection_path.write_text(json.dumps([]), encoding="utf-8")
    denominator = {
        "actual_evaluation_denominator": -1,
        "expected_source_response_count": 1,
        "missing_from_detection_count": 2,
    }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "generated_responses/source.json",
                        "row_count": 1,
                        "model_or_detector_backend": "model",
                    },
                    {
                        "path": "generated_response_detection/model/detailed_outputs.json",
                        "row_count": 0,
                        "detected_response_model_dir": "model",
                        "detected_response_models_observed": ["model"],
                        **denominator,
                    },
                ],
                "totals": {
                    "generated_response_file_count": 1,
                    "generated_response_detection_file_count": 1,
                    "generated_response_rows_total": 1,
                    "generated_response_detection_rows_total": 0,
                },
                "generated_response_detection_denominators": {
                    "model": denominator,
                },
            }
        ),
        encoding="utf-8",
    )

    report = validate_hallucination_artifacts(
        ArtifactSummaryConfig(
            artifact_root=str(root),
            validate_hashes=False,
            scan_public_safety=False,
        ),
        raise_on_error=False,
    )

    assert report.ok is False
    assert any("non-negative" in issue.message for issue in report.errors)


def test_artifact_validation_rejects_physical_files_missing_from_manifest(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "declared.json").write_text(json.dumps([]), encoding="utf-8")
    (root / "unexpected.json").write_text(json.dumps([]), encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "declared.json",
                        "row_count": 0,
                    }
                ],
                "totals": {},
            }
        ),
        encoding="utf-8",
    )

    report = validate_hallucination_artifacts(
        ArtifactSummaryConfig(
            artifact_root=str(root),
            validate_hashes=False,
            validate_counts=False,
            scan_public_safety=False,
        ),
        raise_on_error=False,
    )

    assert report.ok is False
    assert any(
        issue.path == "unexpected.json"
        and "not declared in manifest.files" in issue.message
        for issue in report.errors
    )


def test_artifact_validation_rejects_path_through_symlinked_directory(tmp_path):
    root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.json").write_text(json.dumps([]), encoding="utf-8")
    (root / "escaped").symlink_to(outside, target_is_directory=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "escaped/secret.json",
                        "row_count": 0,
                    }
                ],
                "totals": {},
            }
        ),
        encoding="utf-8",
    )

    report = validate_hallucination_artifacts(
        ArtifactSummaryConfig(
            artifact_root=str(root),
            validate_hashes=False,
            validate_counts=False,
            scan_public_safety=False,
        ),
        raise_on_error=False,
    )

    assert report.ok is False
    assert any(
        "symlink" in issue.message
        for issue in report.errors
    )


def test_validate_and_summarize_scripts_report_expected_counts(capsys):
    assert validate_artifacts.main(["--artifact-root", str(ARTIFACT_ROOT), "--json"]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["ok"] is True
    assert validation["totals"]["generated_response_rows_total"] == 1965
    assert validation["totals"]["generated_response_detection_rows_total"] == 1918

    assert summarize_artifacts.main(
        [
            "--config",
            str(REPO_ROOT / "configs" / "paper" / "hallucination" / "artifact_summary.py"),
            "--artifact-root",
            str(ARTIFACT_ROOT),
            "--json",
        ]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["generated_response_counts"]["loaded_rows_total"] == 1965
    assert summary["generated_response_counts"]["usable_response_rows_total"] == 1964
    assert summary["generated_response_counts"]["unavailable_response_rows_total"] == 1
    assert summary["generated_response_detection"]["loaded_rows_total"] == 1918
    assert summary["native_detector_metrics"]["primary_openai_gpt4o_o3"]["combined_validation"]["counts"]["total"] == 260
    combined_overlap = summary["native_detector_metrics"]["claude_comparative"]["kappa_against_primary_openai"]["combined_overlap"]
    assert combined_overlap["overlap_count"] == 260
    assert combined_overlap["overlap_by_dataset"] == {"stanford": 131, "healthbench": 129}
    assert summary["known_caveats"] == [
        "Generated-response artifacts use Stanford prompt rows.",
        "Per-model detection coverage is recorded by the denominator fields.",
        "Bundle membership is defined by manifest.files.",
        "manifest.json is not self-hashed.",
    ]


def test_drop_policy_uses_evaluated_denominator_and_reports_loss(capsys):
    assert summarize_artifacts.main(
        [
            "--config",
            str(REPO_ROOT / "configs" / "paper" / "hallucination" / "artifact_summary.py"),
            "--artifact-root",
            str(ARTIFACT_ROOT),
            "--uncertain-policy",
            "drop",
            "--json",
        ]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    generated = summary["generated_response_detection"]

    assert generated["raw_rows_total"] == 1918
    assert generated["evaluated_rows_total"] == 1819
    assert generated["dropped_rows_total"] == 99
    assert generated["positive_detection_total"] == 1258
    assert generated["negative_detection_total"] == 561
    combined_overlap = summary["native_detector_metrics"]["claude_comparative"]["kappa_against_primary_openai"]["combined_overlap"]
    assert combined_overlap["left_count"] == 260
    assert combined_overlap["right_count"] == 262
    assert combined_overlap["overlap_count"] == 260
    assert combined_overlap["right_only_by_dataset"]["healthbench"] == 2


def test_sampled_summary_separates_selection_from_artifact_coverage(capsys):
    assert summarize_artifacts.main(
        [
            "--config",
            str(REPO_ROOT / "configs" / "examples" / "hallucination" / "artifact_summary.py"),
            "--artifact-root",
            str(ARTIFACT_ROOT),
            "--json",
        ]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    model = summary["generated_response_detection"]["by_model"]["claude37_positive"]

    assert model["selected_raw_count"] == 2
    assert model["artifact_available_count"] == 131
    assert model["missing_source_count"] == 0
    assert model["coverage_rate"] == 1.0


def test_summarize_all_override_clears_example_sample_cap(capsys):
    assert summarize_artifacts.main(
        [
            "--config",
            str(REPO_ROOT / "configs" / "examples" / "hallucination" / "artifact_summary.py"),
            "--artifact-root",
            str(ARTIFACT_ROOT),
            "--max-samples",
            "all",
            "--json",
        ]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["max_samples_applied"] is None
    assert summary["generated_response_counts"]["loaded_rows_total"] == 1965


def test_loss_aware_summary_accepts_internally_consistent_reduced_bundle(tmp_path):
    root = tmp_path / "reduced"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "files": [],
                "totals": {
                    "generated_response_file_count": 0,
                    "generated_response_detection_file_count": 0,
                    "generated_response_rows_total": 0,
                    "generated_response_detection_rows_total": 0,
                },
                "generated_response_detection_denominators": {},
            }
        ),
        encoding="utf-8",
    )
    config = ArtifactSummaryConfig(
        artifact_root=str(root),
        validate_hashes=False,
        scan_public_safety=False,
    )

    summary = summarize_artifacts.summarize(
        config,
        uncertain_policy="drop",
        include_chi_square=False,
        include_rogan_gladen=False,
    )

    assert summary["artifact_validation"]["ok"] is True
    assert summary["generated_response_counts"]["loaded_rows_total"] == 0
    assert summary["generated_response_detection"]["raw_rows_total"] == 0


def test_offline_scripts_return_nonzero_on_invalid_artifacts(tmp_path, capsys):
    invalid_root = tmp_path / "invalid_artifacts"
    invalid_root.mkdir()
    (invalid_root / "manifest.json").write_text(json.dumps({"files": [], "totals": {}}), encoding="utf-8")

    assert validate_artifacts.main(["--artifact-root", str(invalid_root), "--json"]) == 1
    validate_payload = json.loads(capsys.readouterr().out)
    assert validate_payload["ok"] is False
    assert validate_payload["errors"]

    bad_summary_root = tmp_path / "bad_summary"
    bad_summary_root.mkdir()
    (bad_summary_root / "manifest.json").write_text(json.dumps([]), encoding="utf-8")
    assert summarize_artifacts.main(["--artifact-root", str(bad_summary_root), "--json"]) == 1
    summary_payload = json.loads(capsys.readouterr().out)
    assert "manifest must be a JSON object" in summary_payload["error"]
