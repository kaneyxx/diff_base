"""Tests for configuration loading and validation."""

import pytest
import tempfile
from pathlib import Path
from omegaconf import OmegaConf

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, validate_config, save_config


class TestConfigLoading:
    """Tests for config loading functionality."""

    def test_load_simple_config(self, tmp_path):
        """Test loading a simple YAML config."""
        config_content = """
experiment:
  name: test-experiment
  output_dir: ./outputs

model:
  type: sdxl

training:
  method: lora
  epochs: 10
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)

        assert config.experiment.name == "test-experiment"
        assert config.model.type == "sdxl"
        assert config.training.method == "lora"
        assert config.training.epochs == 10

    def test_config_inheritance(self, tmp_path):
        """Test config inheritance with _base_ key."""
        # Create base config
        base_content = """
model:
  type: sdxl
  dtype: float16

training:
  method: lora
  epochs: 100
"""
        base_path = tmp_path / "base.yaml"
        base_path.write_text(base_content)

        # Create child config that inherits from base
        child_content = f"""
_base_:
  - base.yaml

experiment:
  name: child-experiment

training:
  epochs: 50  # Override epochs
"""
        child_path = tmp_path / "child.yaml"
        child_path.write_text(child_content)

        config = load_config(child_path)

        # Check inherited values
        assert config.model.type == "sdxl"
        assert config.model.dtype == "float16"
        assert config.training.method == "lora"

        # Check overridden values
        assert config.training.epochs == 50

        # Check new values
        assert config.experiment.name == "child-experiment"

    def test_validate_config_success(self):
        """Test validation passes for valid config."""
        config = OmegaConf.create({
            "experiment": {"name": "test"},
            "model": {"type": "sdxl"},
            "training": {"method": "lora"},
        })

        # Should not raise
        validate_config(config)

    def test_validate_config_missing_field(self):
        """Test validation fails for missing required fields."""
        config = OmegaConf.create({
            "experiment": {"name": "test"},
            "model": {"type": "sdxl"},
            # Missing training.method
        })

        with pytest.raises(ValueError, match="Missing required config field"):
            validate_config(config)

    def test_save_config(self, tmp_path):
        """Test saving config to file."""
        config = OmegaConf.create({
            "experiment": {"name": "test"},
            "model": {"type": "sdxl"},
        })

        save_path = tmp_path / "saved.yaml"
        save_config(config, save_path)

        assert save_path.exists()

        # Load and verify
        loaded = load_config(save_path)
        assert loaded.experiment.name == "test"
        assert loaded.model.type == "sdxl"


class TestConfigValues:
    """Tests for specific config value handling."""

    def test_nested_dict_access(self, tmp_path):
        """Test accessing deeply nested config values."""
        config_content = """
model:
  unet:
    in_channels: 4
    attention:
      num_heads: 8
      head_dim: 64
"""
        config_path = tmp_path / "nested.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)

        assert config.model.unet.in_channels == 4
        assert config.model.unet.attention.num_heads == 8
        assert config.model.unet.attention.head_dim == 64

    def test_list_values(self, tmp_path):
        """Test config with list values."""
        config_content = """
training:
  lora:
    target_modules:
      - to_q
      - to_k
      - to_v
    rank: 16
"""
        config_path = tmp_path / "list.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)

        assert len(config.training.lora.target_modules) == 3
        assert "to_q" in config.training.lora.target_modules


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
