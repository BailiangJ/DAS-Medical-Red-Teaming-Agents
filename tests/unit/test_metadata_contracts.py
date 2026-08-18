from copy import deepcopy

from med_red_team.bias.data import build_bias_metadata
from med_red_team.healthbench.data import build_healthbench_metadata
from med_red_team.privacy.data import build_privacy_metadata
from med_red_team.robustness.data import (
    build_robustness_metadata,
    robustness_resume_signature,
)


def _assert_only_current_extra(metadata, removed_aliases):
    assert metadata["run_label"] == "kept"
    for alias in removed_aliases:
        assert alias not in metadata


def test_robustness_metadata_filters_removed_extra_aliases():
    aliases = {
        "source_attack_results": "attack.json",
        "source_testee_model": "legacy-model",
    }
    metadata = build_robustness_metadata(
        phase="attack",
        config={"testee_model": "target", "attacker_strategies": {}},
        target_model="target",
        dataset_info={"path": "dataset.json"},
        source={"dataset_path": "dataset.json"},
        is_partial=False,
        extra={**aliases, "run_label": "kept"},
    )

    _assert_only_current_extra(metadata, aliases)


def test_robustness_replay_signature_binds_both_inputs_and_population():
    metadata = build_robustness_metadata(
        phase="attack_replay",
        config={
            "testee_model": "target",
            "grader_type": "simple",
            "testee_config": {"temperature": 0.0, "max_tokens": 64},
            "testee_system_prompt": "letters only",
            "max_samples": 3,
            "attacker_strategies": {
                "add_none_of_the_above": {"marker": "source-provenance"}
            },
        },
        target_model="target",
        dataset_info={"path": "/tmp/source.jsonl"},
        source={
            "baseline_results_file": "/tmp/baseline.json",
            "baseline_results_sha256": "baseline-sha",
            "attacked_dataset_path": "/tmp/attacks.json",
            "attacked_dataset_sha256": "attacks-sha",
            "dataset_path": "/tmp/source.jsonl",
            "dataset_sha256": "dataset-sha",
        },
        is_partial=True,
        sample_limit=3,
        attack_strategies=["add_none_of_the_above"],
        attack_mode="pre_generated_replay",
        extra={
            "population_policy": "fixed-policy",
            "selected_population": 3,
            "ordered_population_sha256": "population-sha",
        },
    )

    signature = robustness_resume_signature(metadata)

    assert signature["phase"] == "attack_replay"
    assert signature["target"]["model_id"] == "target"
    assert signature["target"]["generation_config"]["max_tokens"] == 64
    assert signature["grader"] == {"type": "simple"}
    assert signature["sampling"] == {"mode": "bounded", "value": 3}
    assert signature["baseline_results_file"] == {"sha256": "baseline-sha"}
    assert signature["attacked_dataset_file"] == {"sha256": "attacks-sha"}
    assert signature["source_identity"] == {"sha256": "dataset-sha"}
    assert signature["attacker_strategies"] == {
        "add_none_of_the_above": {"marker": "source-provenance"}
    }
    assert signature["population_policy"] == "fixed-policy"
    assert signature["selected_population"] == 3
    assert signature["ordered_population_sha256"] == "population-sha"

    changed = deepcopy(metadata)
    changed["models"]["attacker_strategies"]["add_none_of_the_above"][
        "marker"
    ] = "changed"
    assert robustness_resume_signature(changed) != signature


def test_privacy_metadata_filters_removed_extra_aliases():
    aliases = {
        "dataset_source": "legacy-dataset.xlsx",
        "attacked_dataset_source": "legacy-attacked.json",
    }
    metadata = build_privacy_metadata(
        phase="attack",
        config={"testee_model": "target", "attacker_strategies": {}},
        source={"source_results_file": "baseline.json"},
        is_partial=False,
        extra={**aliases, "run_label": "kept"},
    )

    _assert_only_current_extra(metadata, aliases)


def test_bias_metadata_filters_removed_extra_aliases():
    aliases = {
        "attacker_strategies": ["legacy"],
        "dataset_source": "legacy-dataset.xlsx",
        "source_attack_results": "legacy-attack.json",
        "source_testee_model": "legacy-model",
        "attacked_dataset_source": "legacy-attacked.json",
    }
    metadata = build_bias_metadata(
        phase="attack",
        config={"testee_model": "target", "attacker_strategies": {}},
        source={"source_results_file": "baseline.json"},
        is_partial=False,
        extra={**aliases, "run_label": "kept"},
    )

    _assert_only_current_extra(metadata, aliases)


def test_bias_metadata_records_effective_attacker_settings():
    from med_red_team.bias import BiasConfig
    from med_red_team.models import GenerationConfig

    config = BiasConfig(
        attacker_model="default-attacker",
        attacker_config=GenerationConfig(temperature=0.2, max_tokens=123),
        attacker_strategies={
            "language_manipulation": {},
            "cognitive_bias": {"model_id": "override-attacker"},
        },
    )
    metadata = build_bias_metadata(
        phase="attack",
        config=config,
        source={"source_results_file": "baseline.json"},
        is_partial=True,
    )

    assert metadata["models"]["attacker"]["model_id"] == "default-attacker"
    assert metadata["models"]["attack_strategies"]["language_manipulation"]["model_id"] == "default-attacker"
    assert metadata["models"]["attack_strategies"]["language_manipulation"]["config"]["max_tokens"] == 123
    assert metadata["models"]["attack_strategies"]["cognitive_bias"]["model_id"] == "override-attacker"


def test_bias_baseline_metadata_omits_attacker_roles():
    from med_red_team.bias import BiasConfig

    metadata = build_bias_metadata(
        phase="baseline",
        config=BiasConfig(attacker_strategies={"cognitive_bias": {}}),
        source={"data_file": "bias.xlsx"},
        is_partial=False,
    )

    assert set(metadata["models"]) == {"testee", "grader"}
    assert "attack_protocol_version" not in metadata["source"]


def test_bias_attack_metadata_records_protocol_fingerprints():
    from med_red_team.bias import BiasConfig

    metadata = build_bias_metadata(
        phase="attack",
        config=BiasConfig(attacker_strategies={"cognitive_bias": {}}),
        source={"source_results_file": "baseline.json"},
        is_partial=True,
        attack_strategies=["cognitive_bias"],
    )

    assert metadata["source"]["attack_protocol_version"] == "1.0"
    assert metadata["source"]["attack_strategy_registry"]["cognitive_bias"]["class"]
    assert metadata["source"]["attack_prompt_fingerprints"]["cognitive_bias"]["attributes"]
    assert metadata["source"]["attack_prompt_fingerprints"]["cognitive_bias"]["sha256"]
    assert metadata["source"]["attack_implementation_fingerprints"]["bias_attacker"]["sha256"]


def test_healthbench_metadata_accepts_explicit_replay_strategy_config():
    strategy_config = {
        "impossible_measurement": {
            "model_id": "attacker",
            "config": {"temperature": 0.4},
        }
    }
    metadata = build_healthbench_metadata(
        phase="attack_replay_collect",
        config={
            "testee_model": "replay-target",
            "grader_model": "grader",
            "attacker_strategies": {},
        },
        is_partial=False,
        attack_strategies=["impossible_measurement"],
        attack_strategy_config=strategy_config,
    )

    assert metadata["config"]["attacker_strategies"] == {}
    assert metadata["models"]["attack_strategies"] == strategy_config


def test_healthbench_metadata_filters_removed_extra_aliases():
    aliases = {"attacked_dataset_source": "legacy-attacked.json"}
    metadata = build_healthbench_metadata(
        phase="attack",
        config={
            "testee_model": "target",
            "grader_model": "grader",
            "dataset_path": "dataset.jsonl",
            "attacker_strategies": {},
        },
        is_partial=False,
        extra={**aliases, "run_label": "kept"},
    )

    _assert_only_current_extra(metadata, aliases)
