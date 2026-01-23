"""Configuration loading and validation utilities."""

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str | Path) -> DictConfig:
    """Load config file with inheritance support.

    Supports YAML files with `_base_` key for inheriting from other configs.
    Base configs are merged in order, then the current config is merged on top.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Merged configuration as DictConfig.

    Example:
        # configs/experiments/my_experiment.yaml
        _base_:
          - ../models/sdxl.yaml
          - ../training/lora.yaml

        experiment:
          name: "my-experiment"
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = OmegaConf.load(config_path)

    # Handle base config inheritance
    if "_base_" in config:
        base_configs = config.pop("_base_")
        if isinstance(base_configs, str):
            base_configs = [base_configs]

        merged = OmegaConf.create()

        for base_path in base_configs:
            base_full_path = config_path.parent / base_path
            base_config = load_config(base_full_path)
            merged = OmegaConf.merge(merged, base_config)

        config = OmegaConf.merge(merged, config)

    return config


def validate_config(config: DictConfig) -> None:
    """Validate required configuration fields are present.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If required fields are missing.
    """
    required_fields = [
        "experiment.name",
        "model.type",
        "training.method",
    ]

    missing = []
    for field in required_fields:
        if OmegaConf.select(config, field) is None:
            missing.append(field)

    if missing:
        raise ValueError(
            f"Missing required config fields: {missing}. "
            f"Please ensure your config file specifies these fields."
        )

    # Validate model type
    valid_model_types = ["sdxl", "flux"]
    model_type = config.model.type
    if model_type not in valid_model_types:
        raise ValueError(
            f"Invalid model type: {model_type}. "
            f"Must be one of: {valid_model_types}"
        )

    # Validate training method
    valid_methods = [
        "lora", "full_finetune", "dreambooth",
        "controlnet", "textual_inversion", "t2i_adapter"
    ]
    method = config.training.method
    if method not in valid_methods:
        raise ValueError(
            f"Invalid training method: {method}. "
            f"Must be one of: {valid_methods}"
        )


def save_config(config: DictConfig, path: str | Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configs, later configs override earlier ones.

    Args:
        *configs: Configurations to merge.

    Returns:
        Merged configuration.
    """
    return OmegaConf.merge(*configs)


def config_to_dict(config: DictConfig) -> dict[str, Any]:
    """Convert DictConfig to plain Python dict.

    Args:
        config: Configuration to convert.

    Returns:
        Plain dictionary.
    """
    return OmegaConf.to_container(config, resolve=True)


def create_config_from_dict(data: dict[str, Any]) -> DictConfig:
    """Create DictConfig from plain dictionary.

    Args:
        data: Dictionary to convert.

    Returns:
        DictConfig instance.
    """
    return OmegaConf.create(data)


def get_config_value(
    config: DictConfig,
    key: str,
    default: Any = None
) -> Any:
    """Get a nested config value using dot notation.

    Args:
        config: Configuration to query.
        key: Dot-separated key path (e.g., "training.lora.rank").
        default: Default value if key not found.

    Returns:
        Configuration value or default.
    """
    value = OmegaConf.select(config, key)
    return value if value is not None else default
