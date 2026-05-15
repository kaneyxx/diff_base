"""FLUX.1 Model - combines Transformer, VAE, and Text Encoders.

FLUX.1 variants:
- dev: Full model with guidance, 12B parameters
- schnell: Distilled model without guidance, 12B parameters

Image Editing Support:
- Kontext Mode: Reference image editing via encode_reference_images()
- Fill Mode: NOT supported in FLUX.1
"""

from pathlib import Path
from typing import Literal

import torch
from omegaconf import DictConfig
from safetensors.torch import load_file

from ....utils.logging import get_logger
from ...base import BaseDiffusionModel
from .conditioning import (
    create_position_ids,
    prepare_kontext_conditioning,
    rearrange_latent_to_sequence,
)
from .text_encoder import Flux1TextEncoders
from .transformer import Flux1Transformer
from .vae import Flux1VAE
from .weight_mapping import convert_state_dict, detect_format

logger = get_logger(__name__)


class Flux1Model(BaseDiffusionModel):
    """Complete FLUX.1 model with Transformer, VAE, and text encoders.

    Supports both dev and schnell variants.
    """

    def __init__(self, config: DictConfig, variant: str = "dev"):
        """Initialize FLUX.1 model.

        Args:
            config: Model configuration.
            variant: Model variant ("dev" or "schnell").
        """
        self.variant = variant
        super().__init__(config)

    def _build_model(self) -> None:
        """Build all model components."""
        import os
        if os.environ.get("FLUX_TINY_OVERRIDE") == "1":
            self._build_tiny_model()
            return

        # Transformer (DiT)
        transformer_config = self.config.get("transformer", {})
        self.transformer = Flux1Transformer(transformer_config, variant=self.variant)

        # VAE
        vae_config = self.config.get("vae", {})
        self.vae = Flux1VAE(vae_config)

        # Text encoders
        text_config = self.config.get("text_encoder", {})
        self.text_encoders = Flux1TextEncoders(text_config)

        # VAE scaling factors
        self.vae_scale_factor = self.vae.scaling_factor
        self.vae_shift_factor = self.vae.shift_factor

        # Whether to use guidance (dev vs schnell)
        self.use_guidance = self.transformer.guidance_embeds

    def _build_tiny_model(self) -> None:
        """Build tiny stub components for smoke testing (FLUX_TINY_OVERRIDE=1).

        Instantiates a tiny 1-block Flux1Transformer (hidden=64), an identity
        VAE stub, and a random-embedding text encoder stub so the full trainer
        code path can run on CPU without real weights or GPU memory.
        """
        import torch.nn as nn
        from omegaconf import OmegaConf

        logger.info("FLUX_TINY_OVERRIDE=1: building tiny stub model for smoke testing")

        # head_dim = hidden_size // num_heads must equal sum(axes_dims_rope).
        # Default FLUX1_AXES_DIM = (16, 56, 56), sum = 128. Use hidden=128, heads=1 → head_dim=128.
        tiny_cfg = OmegaConf.create({
            "hidden_size": 128,
            "num_attention_heads": 1,
            "num_layers": 1,
            "num_single_layers": 1,
            "in_channels": 64,
            "guidance_embeds": True,
        })
        self.transformer = Flux1Transformer(tiny_cfg, variant="dev")
        self.use_guidance = True

        class _IdentityVAE(nn.Module):
            scaling_factor: float = 1.0
            shift_factor: float = 0.0

            def encode_to_latent(self, image: torch.Tensor) -> torch.Tensor:
                B, _, H, W = image.shape
                return torch.zeros(B, 16, H // 8, W // 8, device=image.device, dtype=image.dtype)

            def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
                B, _, H, W = latent.shape
                return torch.zeros(B, 3, H * 8, W * 8, device=latent.device, dtype=latent.dtype)

        self.vae = _IdentityVAE()
        self.vae_scale_factor = 1.0
        self.vae_shift_factor = 0.0

        class _RandomTextEncoders(nn.Module):
            def encode(self, prompt, device="cpu", **kwargs):
                if isinstance(prompt, str):
                    prompt = [prompt]
                B = len(prompt)
                return {
                    "prompt_embeds": torch.randn(B, 77, 4096, device=device),
                    "pooled_prompt_embeds": torch.randn(B, 768, device=device),
                }

            def freeze(self):
                pass

        self.text_encoders = _RandomTextEncoders()

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor | None = None,
        guidance: torch.Tensor | None = None,
        img_cond_seq: torch.Tensor | None = None,
        img_cond_seq_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through Transformer.

        Args:
            latents: Latent tensor [B, seq_len, channels] or [B, C, H, W].
            timesteps: Timestep values [B].
            encoder_hidden_states: T5 text embeddings [B, txt_seq, dim].
            pooled_projections: CLIP pooled embeddings [B, pool_dim].
            guidance: Optional guidance scale [B].
            img_cond_seq: Kontext conditioning sequence [B, ref_seq, in_channels].
            img_cond_seq_ids: Position IDs for Kontext conditioning [B, ref_seq, 3].
            **kwargs: Additional arguments.

        Returns:
            Predicted output.
        """
        # Flux expects flattened latents [B, seq_len, channels]
        if latents.dim() == 4:
            B, C, H, W = latents.shape  # noqa: N806
            latents = latents.view(B, C, H * W).transpose(1, 2)

        return self.transformer(
            hidden_states=latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            guidance=guidance,
            img_cond_seq=img_cond_seq,
            img_cond_seq_ids=img_cond_seq_ids,
        )

    def encode_reference_images(
        self,
        images: torch.Tensor,
        mode: Literal["kontext"] = "kontext",
        patch_size: int = 2,
    ) -> dict[str, torch.Tensor]:
        """Encode reference images for Kontext conditioning.

        Note: FLUX.1 only supports 'kontext' mode, not 'fill' mode.
        Fill mode (inpainting with channel-wise concatenation) is only
        available in FLUX.2.

        Args:
            images: Reference images [B, 3, H, W] in range [-1, 1].
            mode: Conditioning mode (only 'kontext' is supported).
            patch_size: Patch size for sequence conversion.

        Returns:
            Dictionary with:
            - img_cond_seq: Encoded reference sequence [B, ref_seq, patch_dim]
            - img_cond_seq_ids: Position IDs [B, ref_seq, 3]

        Raises:
            ValueError: If mode is not 'kontext'.
        """
        if mode != "kontext":
            raise ValueError(
                f"FLUX.1 only supports 'kontext' mode, got '{mode}'. "
                "Fill mode is only available in FLUX.2."
            )

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        img_cond_seq, img_cond_seq_ids = prepare_kontext_conditioning(
            reference_images=images,
            vae=self.vae,
            device=device,
            dtype=dtype,
            patch_size=patch_size,
        )

        return {
            "img_cond_seq": img_cond_seq,
            "img_cond_seq_ids": img_cond_seq_ids,
        }

    def encode_text(
        self,
        text: str | list[str],
        device: torch.device | str = "cuda",
    ) -> dict[str, torch.Tensor]:
        """Encode text prompts.

        Args:
            text: Single prompt or list of prompts.
            device: Target device.

        Returns:
            Dictionary with prompt_embeds and pooled_prompt_embeds.
        """
        return self.text_encoders.encode(
            prompt=text,
            device=device,
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space.

        Args:
            image: Image tensor [B, 3, H, W] in range [-1, 1].

        Returns:
            Latent tensor.
        """
        with torch.no_grad():
            latent = self.vae.encode_to_latent(image)
        return latent

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image.

        Args:
            latent: Latent tensor.

        Returns:
            Image tensor [B, 3, H, W] in range [-1, 1].
        """
        with torch.no_grad():
            image = self.vae.decode_from_latent(latent)
        return image

    def _load_diffusers_checkpoint(self, path: Path) -> None:
        """Load from diffusers directory format.

        Args:
            path: Path to diffusers model directory.
        """
        logger.info(f"Loading FLUX.1 from diffusers format: {path}")

        # Load Transformer
        transformer_path = path / "transformer"
        if transformer_path.exists():
            self._load_transformer_weights(transformer_path)
        else:
            logger.warning(f"Transformer not found at {transformer_path}")

        # Load VAE
        vae_path = path / "vae"
        if vae_path.exists():
            self._load_vae_weights(vae_path)
        else:
            logger.warning(f"VAE not found at {vae_path}")

        # Load text encoders
        self.text_encoders.load_pretrained(path)

    def _load_transformer_weights(
        self,
        transformer_path: Path,
        strict: bool = False,
    ) -> None:
        """Load transformer weights from a diffusers directory or single file.

        Auto-detects BFL native vs HF diffusers key format and converts as needed.

        Args:
            transformer_path: Path to transformer directory (diffusers layout) or
                a single .safetensors / .bin file.
            strict: If True, raise on unexpected keys after conversion.
        """
        if transformer_path.is_dir():
            shard_index = transformer_path / "diffusion_pytorch_model.safetensors.index.json"
            single_safe = transformer_path / "diffusion_pytorch_model.safetensors"
            single_bin = transformer_path / "diffusion_pytorch_model.bin"
            if shard_index.exists():
                import json
                with open(shard_index) as f:
                    index = json.load(f)
                shard_files = sorted(set(index["weight_map"].values()))
                raw = {}
                for shard in shard_files:
                    raw.update(load_file(str(transformer_path / shard)))
                weights_file = shard_index
            elif single_safe.exists():
                weights_file = single_safe
                raw = load_file(str(weights_file))
            else:
                weights_file = single_bin
                raw = torch.load(str(weights_file), map_location="cpu")
        else:
            weights_file = transformer_path
            if weights_file.suffix == ".safetensors":
                raw = load_file(str(weights_file))
            else:
                raw = torch.load(str(weights_file), map_location="cpu")

        fmt = detect_format(raw)
        logger.info(f"Detected transformer checkpoint format: '{fmt}'")

        state_dict = convert_state_dict(
            raw,
            source_format=fmt,
            num_heads=self.transformer.num_heads,
            hidden_size=self.transformer.hidden_size,
        )

        missing, unexpected = self.transformer.load_state_dict(state_dict, strict=False)
        logger.info(
            f"Loaded transformer weights — missing: {len(missing)}, unexpected: {len(unexpected)}"
        )
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}"
                           + (" ..." if len(missing) > 5 else ""))
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}"
                           + (" ..." if len(unexpected) > 5 else ""))
            if strict:
                raise KeyError(
                    f"Unexpected keys in checkpoint (strict=True): {unexpected}"
                )

    @classmethod
    def from_bfl_checkpoint(
        cls,
        path: "str | Path",
        variant: str = "dev",
        config: DictConfig | None = None,
        strict: bool = False,
    ) -> "Flux1Model":
        """Load a FLUX.1 model from a BFL native single-file checkpoint.

        Supports the official Black Forest Labs release format
        (e.g., flux1-dev.safetensors, flux1-schnell.safetensors,
        flux1-kontext-dev.safetensors).

        BFL checkpoint key naming differs from HF diffusers; this method
        auto-detects and converts via weight_mapping.convert_state_dict().

        Args:
            path: Path to the .safetensors (or .bin) single-file checkpoint.
            variant: FLUX.1 variant ("dev", "schnell", "kontext"). Default: "dev".
            config: Optional model config DictConfig. If None, uses default
                variant config from Flux1Transformer.VARIANT_CONFIGS.
            strict: If True, raise on unexpected keys after format conversion.

        Returns:
            Initialized Flux1Model with transformer weights loaded.

        Example:
            >>> model = Flux1Model.from_bfl_checkpoint(
            ...     "/path/to/flux1-kontext-dev.safetensors",
            ...     variant="kontext",
            ... )

        Raises:
            ValueError: If checkpoint format cannot be detected.
            KeyError: If strict=True and unexpected keys remain after conversion.
        """
        from omegaconf import OmegaConf

        path = Path(path)
        logger.info(f"Loading FLUX.1 ({variant}) from BFL checkpoint: {path}")

        if config is None:
            variant_cfg = Flux1Transformer.VARIANT_CONFIGS.get(
                variant, Flux1Transformer.VARIANT_CONFIGS["dev"]
            )
            config = OmegaConf.create({"transformer": variant_cfg})

        model = cls(config, variant=variant)
        model._load_transformer_weights(path, strict=strict)
        return model

    def _load_vae_weights(self, vae_path: Path) -> None:
        """Load VAE weights."""
        weights_file = vae_path / "diffusion_pytorch_model.safetensors"
        if not weights_file.exists():
            weights_file = vae_path / "diffusion_pytorch_model.bin"

        if weights_file.suffix == ".safetensors":
            state_dict = load_file(weights_file)
        else:
            state_dict = torch.load(weights_file, map_location="cpu")

        missing, unexpected = self.vae.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"VAE missing keys: {len(missing)}")
        if unexpected:
            logger.warning(f"VAE unexpected keys: {len(unexpected)}")
        logger.info("Loaded VAE weights")

    def compute_image_seq_len(self, latent: torch.Tensor, patch_size: int = 2) -> int:
        """Return the number of image tokens produced by patchifying a latent.

        Args:
            latent: Latent tensor of shape ``[B, C, H, W]``.
            patch_size: Patch size used for patchification (default 2).

        Returns:
            Integer number of image tokens ``(H // patch_size) * (W // patch_size)``.
        """
        _, _, H, W = latent.shape
        return (H // patch_size) * (W // patch_size)

    def prepare_training_inputs(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        text_outputs: dict[str, torch.Tensor],
        guidance_value: float = 1.0,
        batch: dict | None = None,
        patch_size: int = 2,
    ) -> dict[str, torch.Tensor]:
        """Convert latent + text + optional reference into transformer-ready inputs.

        Dispatches between text-to-image and Kontext modes based on whether
        ``batch`` contains a ``"reference_pixel"`` key. Differences between
        FLUX.1 (64-dim patches) and FLUX.2 live in their respective overrides
        of this method, keeping the trainer code version-agnostic.

        Args:
            noisy_latent: Noisy image latent ``[B, C, H, W]``.
            timestep: Per-sample timesteps ``[B]`` in ``[0, 1]``.
            text_outputs: Dict from ``encode_text()`` with keys
                ``"prompt_embeds"`` and ``"pooled_prompt_embeds"``.
            guidance_value: Guidance scale to embed (dev only; ignored for schnell).
            batch: Optional full batch dict. If ``"reference_pixel"`` is present,
                Kontext conditioning is built and included in the output.
            patch_size: Patch size for patchification (default 2).

        Returns:
            Dict of keyword arguments ready to pass to ``model.transformer(**inputs)``.
        """
        B = noisy_latent.shape[0]
        device = noisy_latent.device
        dtype = noisy_latent.dtype

        # Patchify noisy latent [B, C, H, W] → [B, seq, C*patch²]
        hidden_states = rearrange_latent_to_sequence(noisy_latent, patch_size=patch_size)

        # Build target position IDs (stream = 0)
        _, _, H, W = noisy_latent.shape
        h_pat, w_pat = H // patch_size, W // patch_size
        img_ids = create_position_ids(
            batch_size=B,
            height=h_pat,
            width=w_pat,
            device=device,
            dtype=dtype,
            time_offset=0.0,
        )

        # Guidance tensor (dev only)
        guidance: torch.Tensor | None = None
        if self.use_guidance:
            guidance = torch.full(
                (B,), guidance_value, device=device, dtype=dtype
            )

        inputs: dict[str, torch.Tensor] = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": text_outputs["prompt_embeds"],
            "pooled_projections": text_outputs.get("pooled_prompt_embeds"),
            "img_ids": img_ids,
            "guidance": guidance,
        }

        # Kontext conditioning if reference image is provided
        if batch is not None and "reference_pixel" in batch:
            with torch.no_grad():
                img_cond_seq, img_cond_seq_ids = prepare_kontext_conditioning(
                    reference_images=batch["reference_pixel"].to(device, dtype=dtype),
                    vae=self.vae,
                    device=device,
                    dtype=dtype,
                    patch_size=patch_size,
                )
            inputs["img_cond_seq"] = img_cond_seq
            inputs["img_cond_seq_ids"] = img_cond_seq_ids

        return inputs

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing."""
        self.transformer.enable_gradient_checkpointing()
        logger.info("Enabled gradient checkpointing")

    def freeze_vae(self) -> None:
        """Freeze VAE parameters."""
        for param in self.vae.parameters():
            param.requires_grad = False
        logger.info("VAE frozen")

    def freeze_text_encoders(self) -> None:
        """Freeze text encoder parameters."""
        self.text_encoders.freeze()
        logger.info("Text encoders frozen")

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device | str,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Prepare random latents for generation.

        Args:
            batch_size: Batch size.
            height: Image height.
            width: Image width.
            device: Target device.
            dtype: Data type.
            generator: Random generator.

        Returns:
            Random latent tensor.
        """
        # FLUX.1 latent dimensions (16 channels)
        latent_channels = self.vae.latent_channels

        # Calculate latent spatial dimensions
        latent_height = height // 8
        latent_width = width // 8

        shape = (batch_size, latent_channels, latent_height, latent_width)

        latents = torch.randn(
            shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )

        return latents

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Move model to device/dtype."""
        if device is not None or dtype is not None:
            self.transformer = self.transformer.to(device=device, dtype=dtype)
            self.vae = self.vae.to(device=device, dtype=dtype)
            self.text_encoders = self.text_encoders.to(device=device, dtype=dtype)
        return self
