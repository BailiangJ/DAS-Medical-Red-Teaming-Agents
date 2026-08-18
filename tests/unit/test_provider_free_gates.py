from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import med_red_team
from med_red_team import bias, hallucination, healthbench, privacy, robustness
from med_red_team.actors import AttackStrategy
from med_red_team.robustness.attacker import (
    BiasManipulationStrategy,
    GenerateDistractorOptionsStrategy,
)
from scripts import validate_release_tree
from scripts.bias import run_attack as bias_attack
from scripts.bias import run_baseline as bias_baseline
from scripts.privacy import run_attack as privacy_attack
from scripts.privacy import run_baseline as privacy_baseline
from scripts.robustness import run_attack as robustness_attack
from scripts.robustness import run_attack_replay as robustness_attack_replay
from scripts.robustness import run_baseline as robustness_baseline
from scripts.robustness import run_orchestrator as robustness_orchestrator


class BareStrategy(AttackStrategy):
    def apply(self, test_case, context):
        return test_case


def fake_model_pool():
    return SimpleNamespace(get_model=lambda _: SimpleNamespace())


def test_healthbench_package_exports_public_api_lazily():
    assert "HealthBenchConfig" in healthbench.__all__
    assert "HealthBenchRobustnessPipeline" in healthbench.__all__
    assert healthbench.HealthBenchConfig.__name__ == "HealthBenchConfig"
    assert healthbench.HealthBenchRobustnessPipeline.__name__ == "HealthBenchRobustnessPipeline"
    assert healthbench.RubricItem.__name__ == "RubricItem"


def test_root_package_exports_only_top_level_entry_points():
    assert set(med_red_team.__all__) == {
        "GenerationConfig",
        "InfrastructureConfig",
        "ModelFactory",
        "ModelPool",
        "Testee",
    }
    assert "AttackStrategy" not in med_red_team.__all__
    assert "TestCase" not in med_red_team.__all__


def test_axis_packages_keep_public_exports_narrow():
    assert "RobustnessConfig" in robustness.__all__
    assert "RobustnessPipeline" in robustness.__all__

    assert "BiasConfig" in bias.__all__
    assert "BiasPipeline" in bias.__all__
    assert "calculate_vote_entropy" not in bias.__all__
    assert "COGNITIVE_BIAS_TYPES" not in bias.__all__

    assert "PrivacyConfig" in privacy.__all__
    assert "RuleBasedPrivacyGrader" in privacy.__all__
    assert "IDENTIFIERS_TEMPLATE" in privacy.__all__
    assert "validate_privacy_resume_metadata" not in privacy.__all__
    assert "compute_combined_eligibility" not in privacy.__all__

    assert "HealthBenchConfig" in healthbench.__all__
    assert "HEALTHBENCH_GRADER_SYSTEM_PROMPT" not in healthbench.__all__
    assert "validate_healthbench_artifact_metadata" not in healthbench.__all__

    assert "load_detector_outputs" in hallucination.__all__
    assert "load_native_prompt_response_json" in hallucination.__all__
    assert "OrchestratorOutput" not in hallucination.__all__
    assert "normalize_merged_codes" not in hallucination.__all__


def test_attack_strategy_base_serializes_only_shared_fields():
    strategy = BareStrategy(name="bare")
    strategy.num_distractors = 4
    strategy.n_bias_styles = 2

    payload = strategy.to_dict()

    assert "num_distractors" not in payload
    assert "n_bias_styles" not in payload


def test_robustness_strategies_own_axis_specific_serialization():
    distractor = GenerateDistractorOptionsStrategy(
        model_id="fake-attacker",
        model_pool=fake_model_pool(),
        num_distractors=4,
    )
    bias = BiasManipulationStrategy(
        model_id="fake-attacker",
        model_pool=fake_model_pool(),
        n_bias_styles=2,
    )

    assert distractor.to_dict()["num_distractors"] == 4
    assert bias.to_dict()["n_bias_styles"] == 2


def test_privacy_baseline_parser_accepts_hyphenated_options_without_aliases():
    args = privacy_baseline.parse_args(
        [
            "--data-file",
            "cases.xlsx",
            "--prompt-column",
            "prompt",
            "--testee-model",
            "target",
            "--max-samples",
            "2",
        ]
    )

    assert args.data_file == "cases.xlsx"
    assert args.prompt_column == "prompt"
    assert args.testee_model == "target"
    assert args.max_samples == 2

    with pytest.raises(SystemExit):
        privacy_baseline.parse_args(["--data_file", "cases.xlsx"])


def test_privacy_attack_parser_accepts_hyphenated_stage_options_without_aliases():
    args = privacy_attack.parse_args(
        [
            "--baseline-results",
            "baseline.json",
            "--testee-model",
            "target",
            "--grader-max-structured-output-retries",
            "4",
        ]
    )

    assert args.baseline_results == "baseline.json"
    assert args.testee_model == "target"
    assert args.grader_max_structured_output_retries == 4

    with pytest.raises(SystemExit):
        privacy_attack.parse_args(["--baseline_results", "baseline.json"])


def test_bias_parsers_use_hyphenated_options_without_aliases():
    baseline_args = bias_baseline.parse_args(
        [
            "--testee-model",
            "target",
            "--data-file",
            "cases.xlsx",
            "--max-samples",
            "3",
            "--vote-num-baseline",
            "5",
            "--no-cache",
        ]
    )
    assert baseline_args.testee_model == "target"
    assert baseline_args.data_file == "cases.xlsx"
    assert baseline_args.max_samples == 3
    assert baseline_args.vote_num_baseline == 5
    assert baseline_args.no_cache is True

    attack_args = bias_attack.parse_args(["--baseline-results", "baseline.json", "--attacker-model", "attacker"])
    assert attack_args.baseline_results == "baseline.json"
    assert attack_args.attacker_model == "attacker"

    with pytest.raises(SystemExit):
        bias_baseline.parse_args(["--testee_model", "target"])
    with pytest.raises(SystemExit):
        bias_attack.parse_args(["--baseline_results", "baseline.json"])


def test_robustness_parsers_use_hyphenated_options_without_aliases():
    baseline_args = robustness_baseline.parse_args(
        [
            "--testee-model",
            "target",
            "--max-samples",
            "3",
            "--log-dir",
            "logs",
        ]
    )
    assert baseline_args.testee_model == "target"
    assert baseline_args.max_samples == 3
    assert baseline_args.log_dir == "logs"

    attack_args = robustness_attack.parse_args(
        [
            "--baseline-results",
            "baseline.json",
            "--testee-model",
            "target",
            "--max-samples",
            "2",
            "--log-dir",
            "logs",
        ]
    )
    assert attack_args.baseline_results == "baseline.json"
    assert attack_args.testee_model == "target"
    assert attack_args.max_samples == 2
    assert attack_args.log_dir == "logs"

    replay_args = robustness_attack_replay.parse_args(
        [
            "--baseline-results",
            "baseline.json",
            "--attacked-dataset",
            "attacks.json",
            "--max-samples",
            "4",
            "--log-dir",
            "logs",
        ]
    )
    assert replay_args.baseline_results == "baseline.json"
    assert replay_args.attacked_dataset == "attacks.json"
    assert replay_args.max_samples == 4
    assert replay_args.log_dir == "logs"

    orchestrator_args = robustness_orchestrator.parse_args(
        [
            "--baseline-results",
            "baseline.json",
            "--testee-model",
            "target",
            "--orchestrator-model",
            "planner",
            "--tools-model",
            "tools",
            "--max-iterations",
            "5",
            "--max-planner-attempts",
            "2",
            "--max-samples",
            "1",
            "--log-dir",
            "logs",
        ]
    )
    assert orchestrator_args.baseline_results == "baseline.json"
    assert orchestrator_args.testee_model == "target"
    assert orchestrator_args.orchestrator_model == "planner"
    assert orchestrator_args.tools_model == "tools"
    assert orchestrator_args.max_iterations == 5
    assert orchestrator_args.max_planner_attempts == 2
    assert orchestrator_args.max_samples == 1
    assert orchestrator_args.log_dir == "logs"

    with pytest.raises(SystemExit):
        robustness_baseline.parse_args(["--testee_model", "target"])
    with pytest.raises(SystemExit):
        robustness_baseline.parse_args(["--mode", "attacked"])
    with pytest.raises(SystemExit):
        robustness_baseline.parse_args(["--attacked-dataset", "attacks.json"])
    with pytest.raises(SystemExit):
        robustness_attack.parse_args(["--baseline_results", "baseline.json"])
    with pytest.raises(SystemExit):
        robustness_attack.parse_args(["--testee_model", "target"])
    with pytest.raises(SystemExit):
        robustness_attack_replay.parse_args(
            ["--baseline_results", "baseline.json", "--attacked-dataset", "attacks.json"]
        )
    with pytest.raises(SystemExit):
        robustness_attack_replay.parse_args(
            ["--baseline-results", "baseline.json", "--attacked_dataset", "attacks.json"]
        )
    with pytest.raises(SystemExit):
        robustness_orchestrator.parse_args(["--baseline_results", "baseline.json"])
    with pytest.raises(SystemExit):
        robustness_orchestrator.parse_args(["--max_iterations", "5"])


def test_release_tree_validator_rejects_reproducibility_prohibited_directories(
    tmp_path: Path,
):
    prohibited = ("private", "internal", "tmp", "runs")
    for directory_name in prohibited:
        directory = tmp_path / directory_name
        directory.mkdir()
        (directory / "payload.txt").write_text("local run output", encoding="utf-8")

    violations = validate_release_tree.scan_release_tree(tmp_path)
    violation_paths = {violation.path for violation in violations}

    for directory_name in prohibited:
        assert directory_name in violation_paths
        assert f"{directory_name}/payload.txt" in violation_paths


def test_release_tree_validator_rejects_twelve_character_sk_token(tmp_path: Path):
    token = "sk-" + ("a" * 12)
    path = tmp_path / "notes.txt"
    path.write_text(f"credential: {token}\n", encoding="utf-8")

    violations = validate_release_tree.scan_release_tree(tmp_path)

    assert [(violation.path, violation.message) for violation in violations] == [
        ("notes.txt", "contains a credential-like value")
    ]
    assert token not in repr(violations)


def test_release_tree_validator_streams_large_text_for_credentials(tmp_path: Path):
    token = "sk-" + ("b" * 12)
    path = tmp_path / "large.txt"
    with path.open("wb") as handle:
        handle.write(b"prefix\n")
        handle.write(b"x" * (10 * 1024 * 1024))
        handle.write(f"\ncredential: {token}\n".encode("utf-8"))

    violations = validate_release_tree.scan_release_tree(tmp_path)

    assert [(violation.path, violation.message) for violation in violations] == [
        ("large.txt", "contains a credential-like value")
    ]
    assert token not in repr(violations)


def test_release_tree_validator_does_not_decode_unknown_binary_as_text(tmp_path: Path):
    path = tmp_path / "asset.blob"
    path.write_bytes(b"\x00\x01sk-" + (b"c" * 12))

    assert validate_release_tree.scan_release_tree(tmp_path) == []


def test_release_tree_validator_allows_manifest_files_and_rejects_runtime_data(tmp_path: Path):
    artifact_root = tmp_path / "artifacts" / "hallucination"
    artifact_root.mkdir(parents=True)
    manifest = {
        "files": [{"path": "declared.jsonl"}],
    }
    (artifact_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    fake_key = "sk-" + ("0" * 20)
    (artifact_root / "declared.jsonl").write_text(f"{fake_key}\n", encoding="utf-8")

    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "run.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "private.jsonl").write_text("row\n", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=secret", encoding="utf-8")
    private_path = "/" + "home/alice/private"
    (tmp_path / "notes.txt").write_text(f"source: {private_path}\n", encoding="utf-8")

    violations = validate_release_tree.scan_release_tree(
        tmp_path,
        manifest=artifact_root / "manifest.json",
    )
    violation_paths = {violation.path for violation in violations}

    assert "artifacts/hallucination/declared.jsonl" not in violation_paths
    assert "logs" in violation_paths
    assert "model.safetensors" in violation_paths
    assert "private.jsonl" in violation_paths
    assert ".env" in violation_paths
    assert "notes.txt" in violation_paths


def test_release_tree_validator_rejects_allowlisted_symlink_directory(tmp_path: Path):
    artifact_root = tmp_path / "artifacts" / "hallucination"
    outside = tmp_path / "outside"
    artifact_root.mkdir(parents=True)
    outside.mkdir()
    (outside / "secret.json").write_text("{}", encoding="utf-8")
    (artifact_root / "escaped").symlink_to(outside, target_is_directory=True)
    (artifact_root / "manifest.json").write_text(
        json.dumps({"files": [{"path": "escaped/secret.json"}]}),
        encoding="utf-8",
    )

    violations = validate_release_tree.scan_release_tree(
        tmp_path,
        manifest=artifact_root / "manifest.json",
    )

    assert any(
        violation.path == "artifacts/hallucination/escaped"
        and "symlink" in violation.message
        for violation in violations
    )
