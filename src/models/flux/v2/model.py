"""FLUX.2 Model - combines Transformer, VAE, and Text Encoders.

FLUX.2 variants:
- dev: 32B parameters, Mistral-3 text encoder, 32 latent channels
- klein-4b: ~4B parameters, Qwen3-4B text encoder, 128 latent channels
- klein-9b: ~9B parameters, Qwen3-8B text encoder, 128 latent channels

Image Editing Support:
- Kontext Mode: Reference image editing via encode_reference_images()
- Fill Mode: Inpainting via encode_reference_images() with mask
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
    prepare_fill_conditioning,
    prepare_kontext_conditioning,
    rearrange_latent_to_sequence,
)
from .text_encoder import Flux2TextEncoders
from .transformer import Flux2Transformer
from .vae import Flux2VAE

logger = get_logger(__name__)


class Flux2Model(BaseDiffusionModel):
    """Complete FLUX.2 model with Transformer, VAE, and text encoders.

    Supports dev, klein-4b, and klein-9b variants.
    """

    def __init__(self, config: DictConfig, variant: str = "dev"):
        """Initialize FLUX.2 model.

        Args:
            config: Model configuration.
            variant: Model variant ("dev", "klein-4b", or "klein-9b").
        """
        self.variant = variant
        super().__init__(config)

    def _build_model(self) -> None:
        """Build all model components."""
        # Transformer (DiT)
        transformer_config = self.config.get("transformer", {})
        self.transformer = Flux2Transformer(transformer_config, variant=self.variant)

        # VAE
        vae_config = self.config.get("vae", {})
        self.vae = Flux2VAE(vae_config, variant=self.variant)

        # Text encoders
        text_config = self.config.get("text_encoder", {})
        self.text_encoders = Flux2TextEncoders(text_config, variant=self.variant)

        # VAE scaling factors
        self.vae_scale_factor = self.vae.scaling_factor
        self.vae_shift_factor = self.vae.shift_factor

        # All FLUX.2 variants support guidance
        self.use_guidance = self.transformer.guidance_embeds

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor | None = None,
        guidance: torch.Tensor | None = None,
        img_cond_seq: torch.Tensor | None = None,
        img_cond_seq_ids: torch.Tensor | None = None,
        img_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through Transformer.

        Args:
            latents: Latent tensor [B, seq_len, channels] or [B, C, H, W].
            timesteps: Timestep values [B].
            encoder_hidden_states: Text embeddings [B, txt_seq, dim].
            pooled_projections: Pooled text embeddings [B, pool_dim].
            guidance: Optional guidance scale [B].
            img_cond_seq: Kontext conditioning sequence [B, ref_seq, channels].
                Reference image encoded and patchified.
            img_cond_seq_ids: Position IDs for Kontext [B, ref_seq, 3].
            img_cond: Fill conditioning [B, seq, channels + mask_channels].
                Masked image + mask for inpainting.
            **kwargs: Additional arguments.

        Returns:
            Predicted output.
        """
        # FLUX.2 expects flattened latents [B, seq_len, channels]
        if latents.dim() == 4:
            B, C, H, W = latents.shape
            latents = latents.view(B, C, H * W).transpose(1, 2)

        return self.transformer(
            hidden_states=latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            guidance=guidance,
            img_cond_seq=img_cond_seq,
            img_cond_seq_ids=img_cond_seq_ids,
            img_cond=img_cond,
        )

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

    def encode_reference_images(
        self,
        images: torch.Tensor,
        mode: Literal["kontext", "fill"] = "kontext",
        mask: torch.Tensor | None = None,
        patch_size: int = 2,
    ) -> dict[str, torch.Tensor]:
        """Encode reference images for conditioning.

        This method prepares reference images for use with FLUX.2 image editing:

        Kontext Mode:
        - Encodes reference images via VAE
        - Converts to sequence format with position IDs (ref_image_id=1)
        - Returns dict with 'img_cond_seq' and 'img_cond_seq_ids'

        Fill Mode:
        - Applies mask to reference image (keeps unmasked regions)
        - Encodes masked image via VAE
        - Concatenates latent and mask sequences along channel dim
        - Returns dict with 'img_cond'

        Args:
            images: Reference images [B, 3, H, W] in range [-1, 1].
            mode: Editing mode - "kontext" or "fill".
            mask: For Fill mode, binary mask [B, 1, H, W] where 1 = inpaint.
            patch_size: Patch size for sequence conversion (default 2).

        Returns:
            Dictionary containing conditioning tensors:
            - Kontext: {'img_cond_seq': ..., 'img_cond_seq_ids': ...}
            - Fill: {'img_cond': ...}

        Example:
            >>> # Kontext mode
            >>> cond = model.encode_reference_images(ref_images, mode="kontext")
            >>> output = model(latents, timesteps, text_emb, **cond)
            >>>
            >>> # Fill mode
            >>> cond = model.encode_reference_images(ref_images, mode="fill", mask=mask)
            >>> output = model(latents, timesteps, text_emb, **cond)
        """
        device = images.device
        dtype = images.dtype

        if mode == "kontext":
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

        elif mode == "fill":
            if mask is None:
                raise ValueError("Fill mode requires a mask tensor")

            img_cond = prepare_fill_conditioning(
                reference_image=images,
                mask=mask,
                vae=self.vae,
                device=device,
                dtype=dtype,
                patch_size=patch_size,
            )
            return {
                "img_cond": img_cond,
            }

        else:
            raise ValueError(f"Unknown mode: {mode}. Expected 'kontext' or 'fill'.")

    def _load_diffusers_checkpoint(self, path: Path) -> None:
        """Load from diffusers directory format.

        Args:
            path: Path to diffusers model directory.
        """
        logger.info(f"Loading FLUX.2 ({self.variant}) from diffusers format: {path}")

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

        # Load text encoder
        text_encoder_path = path / "text_encoder"
        if text_encoder_path.exists():
            self.text_encoders.load_pretrained(text_encoder_path)
        else:
            # Try loading from HuggingFace
            logger.info("Loading text encoder from HuggingFace...")
            self.text_encoders.load_pretrained()

    def _load_transformer_weights(self, transformer_path: Path) -> None:
        """Load transformer weights."""
        weights_file = transformer_path / "diffusion_pytorch_model.safetensors"
        if not weights_file.exists():
            weights_file = transformer_path / "diffusion_pytorch_model.bin"

        if not weights_file.exists():
            # Try sharded weights
            import glob
            shard_pattern = str(transformer_path / "diffusion_pytorch_model-*.safetensors")
            shards = glob.glob(shard_pattern)
            if shards:
                state_dict = {}
                for shard in sorted(shards):
                    state_dict.update(load_file(shard))
            else:
                logger.error(f"No weights found at {transformer_path}")
                return
        elif weights_file.suffix == ".safetensors":
            state_dict = load_file(weights_file)
        else:
            state_dict = torch.load(weights_file, map_location="cpu")

        missing, unexpected = self.transformer.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Transformer missing keys: {len(missing)}")
        if unexpected:
            logger.warning(f"Transformer unexpected keys: {len(unexpected)}")
        logger.info("Loaded Transformer weights")

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

        Handles FLUX.2-specific differences: larger in_channels (128-512 depending
        on variant), 4D RoPE axes, and optional Fill-mode conditioning.

        Args:
            noisy_latent: Noisy image latent ``[B, C, H, W]``.
            timestep: Per-sample timesteps ``[B]`` in ``[0, 1]``.
            text_outputs: Dict from ``encode_text()`` with keys
                ``"prompt_embeds"`` and ``"pooled_prompt_embeds"``.
            guidance_value: Guidance scale to embed.
            batch: Optional full batch dict. If ``"reference_pixel"`` is present,
                Kontext conditioning is built. If ``"img_cond"`` is present, it is
                passed through directly (Fill mode pre-computed by dataset).
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

        # Guidance tensor
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

        if batch is not None:
            # Kontext mode: encode reference image on-the-fly
            if "reference_pixel" in batch:
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

            # Fill mode: pre-computed by dataset, pass through directly
            elif "img_cond" in batch:
                inputs["img_cond"] = batch["img_cond"].to(device, dtype=dtype)

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
        # FLUX.2 latent dimensions vary by variant
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
