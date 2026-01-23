"""SD3 Triple Text Encoder implementation.

SD3 uses three text encoders:
1. CLIP-L/14 (768 hidden dim) - from OpenAI
2. OpenCLIP bigG/14 (1280 hidden dim) - provides pooled output
3. T5-XXL (4096 hidden dim) - main text encoder for detailed prompts

The encoders work together:
- T5-XXL provides rich text embeddings for cross-attention
- CLIP-L and CLIP-G provide pooled embeddings for conditioning
- Pooled embeddings are concatenated: CLIP-L (768) + CLIP-G (1280) = 2048
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ...utils.logging import get_logger

logger = get_logger(__name__)


class SD3TextEncoders(nn.Module):
    """Triple text encoder for SD3: CLIP-L + OpenCLIP-G + T5-XXL.

    This class manages all three text encoders and provides a unified
    interface for encoding text prompts.
    """

    def __init__(self, config: DictConfig):
        """Initialize SD3 text encoders.

        Args:
            config: Text encoder configuration.
        """
        super().__init__()
        self.config = config

        # Encoders and tokenizers (lazy loaded)
        self.clip_l_encoder = None
        self.clip_l_tokenizer = None

        self.clip_g_encoder = None
        self.clip_g_tokenizer = None

        self.t5_encoder = None
        self.t5_tokenizer = None

        self._loaded = False

        # Max sequence lengths
        self.max_clip_length = config.get("clip_l", {}).get("max_length", 77)
        self.max_t5_length = config.get("t5", {}).get("max_length", 512)

        # Hidden dimensions
        self.clip_l_hidden_size = 768
        self.clip_g_hidden_size = 1280
        self.t5_hidden_size = 4096

    def load_pretrained(self, pretrained_path: str | Path) -> None:
        """Load pretrained text encoders.

        Attempts to load from local path first, then falls back to HuggingFace.

        Args:
            pretrained_path: Path to SD3 model directory or HuggingFace repo.
        """
        from transformers import (
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPTokenizer,
            T5EncoderModel,
            T5Tokenizer,
        )

        pretrained_path = Path(pretrained_path) if isinstance(pretrained_path, str) else pretrained_path

        # === Load CLIP-L ===
        clip_l_path = pretrained_path / "text_encoder" if pretrained_path.is_dir() else None
        if clip_l_path and clip_l_path.exists():
            logger.info(f"Loading CLIP-L from {clip_l_path}")
            self.clip_l_encoder = CLIPTextModelWithProjection.from_pretrained(clip_l_path)
            self.clip_l_tokenizer = CLIPTokenizer.from_pretrained(clip_l_path)
        else:
            logger.info("Loading CLIP-L from openai/clip-vit-large-patch14")
            self.clip_l_encoder = CLIPTextModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            self.clip_l_tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )

        # === Load OpenCLIP-G (CLIP bigG) ===
        clip_g_path = pretrained_path / "text_encoder_2" if pretrained_path.is_dir() else None
        if clip_g_path and clip_g_path.exists():
            logger.info(f"Loading CLIP-G from {clip_g_path}")
            self.clip_g_encoder = CLIPTextModelWithProjection.from_pretrained(clip_g_path)
            self.clip_g_tokenizer = CLIPTokenizer.from_pretrained(clip_g_path)
        else:
            # Note: OpenCLIP bigG may need different loading
            logger.info("Loading CLIP-G from laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
            try:
                self.clip_g_encoder = CLIPTextModelWithProjection.from_pretrained(
                    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
                )
                self.clip_g_tokenizer = CLIPTokenizer.from_pretrained(
                    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
                )
            except Exception as e:
                logger.warning(f"Could not load CLIP-G: {e}. Using CLIP-L as fallback.")
                self.clip_g_encoder = self.clip_l_encoder
                self.clip_g_tokenizer = self.clip_l_tokenizer

        # === Load T5-XXL ===
        t5_path = pretrained_path / "text_encoder_3" if pretrained_path.is_dir() else None
        if t5_path and t5_path.exists():
            logger.info(f"Loading T5-XXL from {t5_path}")
            self.t5_encoder = T5EncoderModel.from_pretrained(t5_path)
            self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_path)
        else:
            logger.info("Loading T5-XXL from google/t5-v1_1-xxl")
            self.t5_encoder = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl")
            self.t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")

        self._loaded = True
        logger.info("All text encoders loaded successfully")

    def encode(
        self,
        prompt: str | list[str],
        device: torch.device | str = "cuda",
        num_images_per_prompt: int = 1,
        max_t5_length: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Encode text prompts using all three encoders.

        Args:
            prompt: Single prompt or list of prompts.
            device: Target device.
            num_images_per_prompt: Number of images per prompt (for repeating).
            max_t5_length: Override max T5 sequence length.

        Returns:
            Dictionary containing:
                - prompt_embeds: T5 text embeddings [B, seq_len, 4096]
                - pooled_prompt_embeds: Concatenated CLIP pooled [B, 2048]
                - clip_l_embeds: CLIP-L hidden states [B, 77, 768]
                - clip_g_embeds: CLIP-G hidden states [B, 77, 1280]
        """
        if not self._loaded:
            raise RuntimeError(
                "Text encoders not loaded. Call load_pretrained() first."
            )

        if isinstance(prompt, str):
            prompt = [prompt]

        batch_size = len(prompt)
        max_t5_length = max_t5_length or self.max_t5_length

        # === Encode with CLIP-L ===
        clip_l_tokens = self.clip_l_tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_clip_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            clip_l_output = self.clip_l_encoder(
                clip_l_tokens.input_ids.to(device),
                attention_mask=clip_l_tokens.attention_mask.to(device),
                output_hidden_states=True,
            )
            clip_l_embeds = clip_l_output.hidden_states[-2]  # Penultimate layer
            clip_l_pooled = clip_l_output.text_embeds  # Pooled output

        # === Encode with CLIP-G ===
        clip_g_tokens = self.clip_g_tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_clip_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            clip_g_output = self.clip_g_encoder(
                clip_g_tokens.input_ids.to(device),
                attention_mask=clip_g_tokens.attention_mask.to(device),
                output_hidden_states=True,
            )
            clip_g_embeds = clip_g_output.hidden_states[-2]
            clip_g_pooled = clip_g_output.text_embeds

        # === Encode with T5-XXL ===
        t5_tokens = self.t5_tokenizer(
            prompt,
            padding="max_length",
            max_length=max_t5_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            t5_output = self.t5_encoder(
                t5_tokens.input_ids.to(device),
                attention_mask=t5_tokens.attention_mask.to(device),
            )
            t5_embeds = t5_output.last_hidden_state

        # === Combine pooled embeddings ===
        # CLIP-L pooled (768) + CLIP-G pooled (1280) = 2048
        pooled_prompt_embeds = torch.cat([clip_l_pooled, clip_g_pooled], dim=-1)

        # Repeat for num_images_per_prompt
        if num_images_per_prompt > 1:
            t5_embeds = t5_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            clip_l_embeds = clip_l_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            clip_g_embeds = clip_g_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        return {
            "prompt_embeds": t5_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "clip_l_embeds": clip_l_embeds,
            "clip_g_embeds": clip_g_embeds,
            "t5_attention_mask": t5_tokens.attention_mask.to(device),
        }

    def encode_negative(
        self,
        negative_prompt: str | list[str] | None,
        batch_size: int,
        device: torch.device | str = "cuda",
    ) -> dict[str, torch.Tensor]:
        """Encode negative prompts (or create empty embeddings).

        Args:
            negative_prompt: Negative prompt(s) or None for empty.
            batch_size: Batch size for empty embeddings.
            device: Target device.

        Returns:
            Dictionary with same keys as encode().
        """
        if negative_prompt is None or negative_prompt == "" or negative_prompt == [""]:
            # Create zero embeddings
            return {
                "prompt_embeds": torch.zeros(
                    batch_size, self.max_t5_length, self.t5_hidden_size,
                    device=device, dtype=torch.float32
                ),
                "pooled_prompt_embeds": torch.zeros(
                    batch_size, self.clip_l_hidden_size + self.clip_g_hidden_size,
                    device=device, dtype=torch.float32
                ),
            }

        return self.encode(negative_prompt, device=device)

    def to(
        self,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Move encoders to device/dtype.

        Args:
            device: Target device.
            dtype: Target dtype.

        Returns:
            Self for chaining.
        """
        if self.clip_l_encoder is not None:
            self.clip_l_encoder = self.clip_l_encoder.to(device=device, dtype=dtype)
        if self.clip_g_encoder is not None:
            self.clip_g_encoder = self.clip_g_encoder.to(device=device, dtype=dtype)
        if self.t5_encoder is not None:
            self.t5_encoder = self.t5_encoder.to(device=device, dtype=dtype)
        return self

    def freeze(self) -> None:
        """Freeze all text encoder parameters."""
        for encoder in [self.clip_l_encoder, self.clip_g_encoder, self.t5_encoder]:
            if encoder is not None:
                for param in encoder.parameters():
                    param.requires_grad = False
        logger.info("Text encoders frozen")

    def unfreeze(self) -> None:
        """Unfreeze all text encoder parameters."""
        for encoder in [self.clip_l_encoder, self.clip_g_encoder, self.t5_encoder]:
            if encoder is not None:
                for param in encoder.parameters():
                    param.requires_grad = True

    def parameters(self, recurse: bool = True):
        """Get all parameters from all encoders."""
        params = []
        for encoder in [self.clip_l_encoder, self.clip_g_encoder, self.t5_encoder]:
            if encoder is not None:
                params.extend(encoder.parameters(recurse))
        return iter(params)

    def get_param_count(self) -> dict:
        """Get parameter counts for each encoder."""
        counts = {}
        for name, encoder in [
            ("clip_l", self.clip_l_encoder),
            ("clip_g", self.clip_g_encoder),
            ("t5", self.t5_encoder),
        ]:
            if encoder is not None:
                counts[name] = sum(p.numel() for p in encoder.parameters())
        counts["total"] = sum(counts.values())
        return counts
