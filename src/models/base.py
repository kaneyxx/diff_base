"""Base classes for diffusion models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from safetensors.torch import load_file

from ..utils.logging import get_logger

logger = get_logger(__name__)


class BaseDiffusionModel(ABC, nn.Module):
    """Abstract base class for diffusion models.

    All diffusion model architectures should inherit from this class
    and implement the required abstract methods.
    """

    def __init__(self, config: DictConfig):
        """Initialize the diffusion model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self._gradient_checkpointing = False
        self._build_model()

    @abstractmethod
    def _build_model(self) -> None:
        """Build model architecture from config.

        This method should create all model components based on
        the configuration. Called during __init__.
        """
        pass

    @abstractmethod
    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for noise prediction.

        Args:
            latents: Noisy latent representations [B, C, H, W].
            timesteps: Timestep values [B].
            encoder_hidden_states: Text encoder outputs [B, seq_len, dim].
            **kwargs: Additional model-specific arguments.

        Returns:
            Predicted noise or velocity [B, C, H, W].
        """
        pass

    @abstractmethod
    def encode_text(
        self,
        text: str | list[str],
        device: torch.device | str = "cuda",
    ) -> dict[str, torch.Tensor]:
        """Encode text prompts to embeddings.

        Args:
            text: Single prompt or list of prompts.
            device: Device to place embeddings on.

        Returns:
            Dictionary containing text embeddings and any pooled outputs.
        """
        pass

    @abstractmethod
    def encode_image(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Encode image to latent space using VAE.

        Args:
            image: Image tensor [B, 3, H, W] in range [-1, 1].

        Returns:
            Latent representation [B, C, h, w].
        """
        pass

    @abstractmethod
    def decode_latent(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent to image using VAE.

        Args:
            latent: Latent tensor [B, C, h, w].

        Returns:
            Decoded image [B, 3, H, W] in range [-1, 1].
        """
        pass

    def load_pretrained(self, checkpoint_path: str | Path) -> None:
        """Load pretrained weights into the model.

        Supports multiple formats:
        - Diffusers directory format
        - Single safetensors file
        - PyTorch checkpoint

        Args:
            checkpoint_path: Path to checkpoint.
        """
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_dir():
            self._load_diffusers_checkpoint(checkpoint_path)
        elif checkpoint_path.suffix == ".safetensors":
            self._load_safetensors(checkpoint_path)
        elif checkpoint_path.suffix in (".pt", ".pth", ".bin"):
            self._load_pytorch_checkpoint(checkpoint_path)
        else:
            raise ValueError(
                f"Unknown checkpoint format: {checkpoint_path}. "
                f"Supported: directory (diffusers), .safetensors, .pt/.pth/.bin"
            )

    def _load_diffusers_checkpoint(self, path: Path) -> None:
        """Load from diffusers directory format.

        Args:
            path: Path to diffusers model directory.
        """
        # Subclasses should override this for architecture-specific loading
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _load_diffusers_checkpoint"
        )

    def _load_safetensors(self, path: Path) -> None:
        """Load from safetensors file.

        Args:
            path: Path to safetensors file.
        """
        state_dict = load_file(path)
        self._load_state_dict_flexible(state_dict)

    def _load_pytorch_checkpoint(self, path: Path) -> None:
        """Load from PyTorch checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        state_dict = torch.load(path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self._load_state_dict_flexible(state_dict)

    def _load_state_dict_flexible(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load state dict with flexible key matching.

        Args:
            state_dict: State dictionary to load.
        """
        # Map keys if needed
        mapped_state_dict = self._map_checkpoint_keys(state_dict)

        # Load with non-strict mode to handle missing/extra keys
        missing, unexpected = self.load_state_dict(mapped_state_dict, strict=False)

        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

        logger.info(f"Loaded {len(mapped_state_dict)} parameters")

    def _map_checkpoint_keys(
        self,
        state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map checkpoint keys to model keys.

        Override this method to handle key name differences between
        checkpoint format and model architecture.

        Args:
            state_dict: Original state dictionary.

        Returns:
            State dictionary with remapped keys.
        """
        return state_dict

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        self._gradient_checkpointing = True

        # Enable on all submodules that support it
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
            if hasattr(module, "_gradient_checkpointing"):
                module._gradient_checkpointing = True

        logger.info("Gradient checkpointing enabled")

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False
            if hasattr(module, "_gradient_checkpointing"):
                module._gradient_checkpointing = False

    @property
    def device(self) -> torch.device:
        """Get the device of model parameters."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of model parameters."""
        return next(self.parameters()).dtype

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Get list of trainable parameters.

        Returns:
            List of parameters with requires_grad=True.
        """
        return [p for p in self.parameters() if p.requires_grad]

    def freeze(self) -> None:
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_param_count(self) -> dict[str, int]:
        """Get parameter counts.

        Returns:
            Dictionary with total and trainable parameter counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
