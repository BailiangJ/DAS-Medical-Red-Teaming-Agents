"""
HealthBench Evaluation Configuration
=====================================

Configuration management for HealthBench evaluation and robustness testing.

Example:
    >>> from med_red_team.healthbench.config import HealthBenchConfig
    >>>
    >>> # Baseline config
    >>> config = HealthBenchConfig(
    ...     testee_model="gpt-4o",
    ...     grader_model="gemini-2.5-flash",
    ...     dataset_path="test/healthbench_dataset/consensus.jsonl",
    ...     max_samples=100
    ... )
    >>>
    >>> # Robustness config with attacks
    >>> config = HealthBenchConfig(
    ...     testee_model="gpt-4o",
    ...     grader_model="gemini-2.5-flash",
    ...     attacker_strategies={
    ...         "distraction": {
    ...             "model_id": "claude-sonnet-4-20250514",
    ...             "config": GenerationConfig(temperature=1.0)
    ...         }
    ...     },
    ...     max_samples=50
    ... )
    >>>
    >>> # Save/load
    >>> config.to_json("configs/my_config.json")
    >>> config = HealthBenchConfig.from_json("configs/my_config.json")
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional

from med_red_team.models import GenerationConfig
from med_red_team.config import BaseConfig


@dataclass
class HealthBenchConfig(BaseConfig):
    """
    Configuration for HealthBench evaluation and robustness testing.

    Attributes:
        # Model IDs
        testee_model: Model ID for the model being tested
        grader_model: Model ID for rubric grading

        # Generation configs
        testee_config: Generation config for testee
        grader_config: Generation config for grader (usually deterministic)

        # System prompts
        testee_system_prompt: System prompt for testee

        # Data settings
        dataset_path: Path to HealthBench JSONL file
        max_samples: Maximum number of samples to evaluate (None = all)
        filter_single_turn: Whether to filter to single-turn conversations only

        # Attack strategies (for robustness testing)
        attacker_strategies: Dict mapping strategy names to their __init__ kwargs
                          e.g., {"distraction": {"model_id": "...", "config": ...}}
                          Empty dict for baseline-only evaluation

        # Output settings
        log_dir: Directory for saving results
    """

    # Model IDs
    testee_model: str = "gpt-4o"
    grader_model: str = "gemini-2.5-flash"

    # Generation configs
    testee_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(
            temperature=0.0,
            max_tokens=2048
        )
    )
    grader_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(
            temperature=0.0,
            max_tokens=4096
        )
    )

    # System prompts
    testee_system_prompt: str = "You are a helpful medical assistant."

    # Data settings
    dataset_path: Optional[str] = None
    max_samples: Optional[int] = None  # None = all samples
    filter_single_turn: bool = True  # Filter to single-turn conversations

    # Attack strategies (empty dict for baseline-only)
    # Example: {
    #     "distraction": {"model_id": "claude-sonnet-4-20250514", "config": GenerationConfig(...)},
    #     "impossible_measurement": {"model_id": "claude-sonnet-4-20250514"}
    # }
    attacker_strategies: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Output settings
    log_dir: str = "logs/healthbench"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary for JSON serialization.

        Handles GenerationConfig serialization.
        """
        # Serialize attacker_strategies using base class method
        attacker_strategies_dict = self.serialize_attacker_strategies(self.attacker_strategies)

        config_dict = {
            "testee_model": self.testee_model,
            "grader_model": self.grader_model,
            "testee_config": asdict(self.testee_config),
            "grader_config": asdict(self.grader_config),
            "testee_system_prompt": self.testee_system_prompt,
            "dataset_path": self.dataset_path,
            "max_samples": self.max_samples,
            "filter_single_turn": self.filter_single_turn,
            "attacker_strategies": attacker_strategies_dict,
            "log_dir": self.log_dir
        }

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "HealthBenchConfig":
        """
        Create config from dictionary (e.g., from JSON).

        Handles GenerationConfig deserialization.
        """
        config_dict = config_dict.copy()

        # Convert GenerationConfig dicts back to objects
        for key in ['testee_config', 'grader_config']:
            if key in config_dict and isinstance(config_dict[key], dict):
                config_dict[key] = GenerationConfig(**config_dict[key])

        # Deserialize attacker_strategies using base class method
        if "attacker_strategies" in config_dict and isinstance(config_dict["attacker_strategies"], dict):
            config_dict["attacker_strategies"] = cls.deserialize_attacker_strategies(
                config_dict["attacker_strategies"]
            )

        return cls(**config_dict)
