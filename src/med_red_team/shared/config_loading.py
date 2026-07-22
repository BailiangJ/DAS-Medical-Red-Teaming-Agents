"""Explicit loading for axis-owned Python configuration presets."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import TypeVar


ConfigT = TypeVar("ConfigT")


def _load_module(path: Path) -> ModuleType:
    module_name = f"med_red_team_user_config_{abs(hash(path.resolve()))}"
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load configuration module: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(path: str | Path, expected_type: type[ConfigT]) -> ConfigT:
    """Load the single public ``CONFIG`` object from a Python preset file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    module = _load_module(config_path)
    if not hasattr(module, "CONFIG"):
        raise ValueError(f"Configuration file must export CONFIG: {config_path}")

    config = module.CONFIG
    if not isinstance(config, expected_type):
        raise TypeError(
            f"CONFIG in {config_path} must be {expected_type.__name__}, "
            f"got {type(config).__name__}"
        )
    return config
