"""Configuration objects for hallucination v2 workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from med_red_team.config import BaseConfig
from med_red_team.models import GenerationConfig
from med_red_team.shared.config_loading import load_config as load_python_config
from med_red_team.shared.io import read_json, to_jsonable


MaxSamples = int | None

DEFAULT_DETECTOR_MODELS = {
    "openai": {"orchestrator_model": "gpt-4o", "sub_agent_model": "gpt-4o"},
    "claude": {
        "orchestrator_model": "claude-sonnet-4-20250514",
        "sub_agent_model": "inherit",
    },
}


def validate_max_samples(max_samples: MaxSamples) -> None:
    """Accept None for all rows, zero for no work, or a positive row cap."""
    if isinstance(max_samples, bool) or (
        max_samples is not None and not isinstance(max_samples, int)
    ):
        raise ValueError("max_samples must be None, 0, or a positive integer")
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be None, 0, or a positive integer")


def resolve_max_samples(max_samples: MaxSamples, total: int) -> int:
    """Resolve public v2 None/0/positive semantics against available rows."""
    validate_max_samples(max_samples)
    if total < 0:
        raise ValueError("total must be non-negative")
    if max_samples is None:
        return total
    return min(max_samples, total)


def _generation_config_to_dict(config: GenerationConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, GenerationConfig):
        return asdict(config)
    return dict(config)


def _generation_config_from_value(value: GenerationConfig | dict[str, Any]) -> GenerationConfig:
    if isinstance(value, GenerationConfig):
        return value
    if isinstance(value, dict):
        return GenerationConfig(**value)
    raise TypeError(f"Expected GenerationConfig or dict, got {type(value).__name__}")


@dataclass
class StanfordResponseGenerationConfig(BaseConfig):
    """Configuration for generating responses from a prompt dataset."""

    dataset_path: str = "artifacts/hallucination/datasets/stanford_redteaming_131_positive_cases.json"
    output_dir: str = "logs/hallucination/generated_responses"
    model_ids: list[str] = field(default_factory=lambda: ["gpt-4o"])
    system_prompt: str = "You are a helpful medical assistant."
    generation_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(temperature=0.7, max_tokens=4096)
    )
    max_samples: MaxSamples = None
    prompt_key: str = "Prompt"
    response_key: str = "Response"
    row_id_key: str = "row_idx"
    overwrite: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_max_samples(self.max_samples)
        if not self.model_ids:
            raise ValueError("model_ids must contain at least one model id")
        source_keys = {
            "prompt_key": self.prompt_key,
            "response_key": self.response_key,
            "row_id_key": self.row_id_key,
        }
        for name, value in source_keys.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if len(set(source_keys.values())) != len(source_keys):
            raise ValueError("prompt_key, response_key, and row_id_key must be distinct")
        self.generation_config = _generation_config_from_value(self.generation_config)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["generation_config"] = _generation_config_to_dict(self.generation_config)
        return to_jsonable(data)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "StanfordResponseGenerationConfig":
        data = dict(config_dict)
        if "generation_config" in data:
            data["generation_config"] = _generation_config_from_value(data["generation_config"])
        return cls(**data)

    def resolved_sample_count(self, total: int) -> int:
        return resolve_max_samples(self.max_samples, total)


@dataclass
class DetectorExecutionConfig(BaseConfig):
    """Config for explicit hallucination detector runs."""

    input_path: str
    output_path: str
    backend: Literal["openai", "claude"] = "openai"
    input_format: Literal["auto", "native", "generated"] = "auto"
    prompt_key: str = "Prompt"
    response_key: str = "Response"
    row_id_key: str = "row_idx"
    orchestrator_model: str = ""
    sub_agent_model: str = ""
    search_agent_model: str = "o3"
    max_samples: MaxSamples = None
    start_idx: int = 0
    end_idx: int | None = None
    ignore_existing: bool = False
    overwrite: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_max_samples(self.max_samples)
        if self.backend not in {"openai", "claude"}:
            raise ValueError("backend must be 'openai' or 'claude'")
        if self.input_format not in {"auto", "native", "generated"}:
            raise ValueError("input_format must be 'auto', 'native', or 'generated'")
        source_keys = {
            "prompt_key": self.prompt_key,
            "response_key": self.response_key,
            "row_id_key": self.row_id_key,
        }
        for name, value in source_keys.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if len(set(source_keys.values())) != len(source_keys):
            raise ValueError("prompt_key, response_key, and row_id_key must be distinct")
        backend_defaults = DEFAULT_DETECTOR_MODELS[self.backend]
        if not self.orchestrator_model.strip():
            self.orchestrator_model = backend_defaults["orchestrator_model"]
        if not self.sub_agent_model.strip():
            self.sub_agent_model = backend_defaults["sub_agent_model"]
        if self.start_idx < 0:
            raise ValueError("start_idx must be non-negative")
        if self.end_idx is not None and self.end_idx < self.start_idx:
            raise ValueError("end_idx must be >= start_idx")

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "DetectorExecutionConfig":
        return cls(**dict(config_dict))

    def resolve_row_slice(self, total: int) -> slice:
        """Return the requested row slice after start/end/max_samples semantics."""
        if total < 0:
            raise ValueError("total must be non-negative")
        stop = total if self.end_idx is None else min(self.end_idx + 1, total)
        if self.start_idx > stop:
            stop = self.start_idx
        requested = max(0, stop - self.start_idx)
        limit = resolve_max_samples(self.max_samples, requested)
        return slice(self.start_idx, self.start_idx + limit)


@dataclass
class ArtifactSummaryConfig(BaseConfig):
    """Config for offline validation/summarization of packaged artifacts."""

    artifact_root: str = "artifacts/hallucination"
    manifest_name: str = "manifest.json"
    validate_hashes: bool = True
    validate_json: bool = True
    validate_counts: bool = True
    scan_public_safety: bool = True
    raise_on_error: bool = False
    max_samples: MaxSamples = None
    expected_stanford_current_rows: int = 131
    expected_healthbench_current_rows: int = 131
    expected_generated_response_file_count: int = 15
    expected_generated_response_detection_file_count: int = 15
    expected_generated_response_rows_total: int = 1965
    expected_generated_response_detection_rows_total: int = 1918

    def __post_init__(self) -> None:
        validate_max_samples(self.max_samples)
        expected_values = {
            "expected_stanford_current_rows": self.expected_stanford_current_rows,
            "expected_healthbench_current_rows": self.expected_healthbench_current_rows,
            "expected_generated_response_file_count": self.expected_generated_response_file_count,
            "expected_generated_response_detection_file_count": self.expected_generated_response_detection_file_count,
            "expected_generated_response_rows_total": self.expected_generated_response_rows_total,
            "expected_generated_response_detection_rows_total": self.expected_generated_response_detection_rows_total,
        }
        for name, value in expected_values.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def manifest_path(self) -> Path:
        return Path(self.artifact_root) / self.manifest_name

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ArtifactSummaryConfig":
        return cls(**dict(config_dict))


CONFIG_CLASS_BY_NAME = {
    "stanford_response_generation": StanfordResponseGenerationConfig,
    "detector_execution": DetectorExecutionConfig,
    "artifact_summary": ArtifactSummaryConfig,
}


def load_config(path: str | Path, config_type: type[Any]) -> Any:
    """Load JSON configs or Python files exporting a public CONFIG object."""
    config_path = Path(path)
    if config_path.suffix == ".py":
        return load_python_config(config_path, config_type)
    if not hasattr(config_type, "from_dict"):
        raise TypeError("config_type must provide from_dict")
    data = read_json(config_path)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration JSON must contain an object: {config_path}")
    return config_type.from_dict(data)


__all__ = [
    "ArtifactSummaryConfig",
    "CONFIG_CLASS_BY_NAME",
    "DEFAULT_DETECTOR_MODELS",
    "DetectorExecutionConfig",
    "StanfordResponseGenerationConfig",
    "load_config",
    "resolve_max_samples",
    "validate_max_samples",
]
