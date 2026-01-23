"""Flux Text Encoders (T5 and CLIP-L)."""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig


class FluxTextEncoders(nn.Module):
    """Flux dual text encoders (T5-XXL and CLIP-L).

    Uses HuggingFace transformers for the actual models.
    """

    def __init__(self, config: DictConfig):
        """Initialize Flux text encoders.

        Args:
            config: Text encoder configuration.
        """
        super().__init__()
        self.config = config

        # Will be loaded from pretrained
        self.t5_encoder = None
        self.clip_encoder = None
        self.t5_tokenizer = None
        self.clip_tokenizer = None

        self._loaded = False
        self.max_t5_length = config.get("t5", {}).get("max_position_embeddings", 512)
        self.max_clip_length = config.get("clip_l", {}).get("max_position_embeddings", 77)

    def load_pretrained(self, pretrained_path: str | Path) -> None:
        """Load pretrained text encoders.

        Args:
            pretrained_path: Path to Flux model directory.
        """
        from transformers import (
            T5EncoderModel,
            T5Tokenizer,
            CLIPTextModel,
            CLIPTokenizer,
        )

        pretrained_path = Path(pretrained_path)

        # T5-XXL encoder
        t5_path = pretrained_path / "text_encoder"
        if t5_path.exists():
            self.t5_encoder = T5EncoderModel.from_pretrained(t5_path)
            self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_path)
        else:
            # Try loading from HuggingFace
            self.t5_encoder = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl")
            self.t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")

        # CLIP-L encoder
        clip_path = pretrained_path / "text_encoder_2"
        if clip_path.exists():
            self.clip_encoder = CLIPTextModel.from_pretrained(clip_path)
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_path)
        else:
            self.clip_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )

        self._loaded = True

    def encode(
        self,
        prompt: str | list[str],
        device: torch.device | str = "cuda",
        max_t5_length: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Encode text prompts.

        Args:
            prompt: Text prompt or list of prompts.
            device: Target device.
            max_t5_length: Maximum T5 sequence length.

        Returns:
            Dictionary with:
                - prompt_embeds: T5 text embeddings
                - pooled_prompt_embeds: CLIP pooled embeddings
        """
        if not self._loaded:
            raise RuntimeError("Text encoders not loaded. Call load_pretrained first.")

        if isinstance(prompt, str):
            prompt = [prompt]

        max_t5_length = max_t5_length or self.max_t5_length

        # Encode with T5
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

        # Encode with CLIP (for pooled embeddings)
        clip_tokens = self.clip_tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_clip_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            clip_output = self.clip_encoder(
                clip_tokens.input_ids.to(device),
                attention_mask=clip_tokens.attention_mask.to(device),
            )
            pooled_embeds = clip_output.pooler_output

        return {
            "prompt_embeds": t5_embeds,
            "pooled_prompt_embeds": pooled_embeds,
            "attention_mask": t5_tokens.attention_mask.to(device),
        }

    def forward(
        self,
        t5_input_ids: torch.Tensor,
        clip_input_ids: torch.Tensor,
        t5_attention_mask: Optional[torch.Tensor] = None,
        clip_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with pre-tokenized inputs.

        Args:
            t5_input_ids: T5 token IDs.
            clip_input_ids: CLIP token IDs.
            t5_attention_mask: T5 attention mask.
            clip_attention_mask: CLIP attention mask.

        Returns:
            Tuple of (t5_embeds, pooled_embeds).
        """
        if not self._loaded:
            raise RuntimeError("Text encoders not loaded.")

        # T5 encoding
        t5_output = self.t5_encoder(
            t5_input_ids,
            attention_mask=t5_attention_mask,
        )
        t5_embeds = t5_output.last_hidden_state

        # CLIP encoding
        clip_output = self.clip_encoder(
            clip_input_ids,
            attention_mask=clip_attention_mask,
        )
        pooled_embeds = clip_output.pooler_output

        return t5_embeds, pooled_embeds

    def to(self, device: torch.device | str, dtype: Optional[torch.dtype] = None):
        """Move encoders to device."""
        if self.t5_encoder is not None:
            self.t5_encoder = self.t5_encoder.to(device, dtype=dtype)
        if self.clip_encoder is not None:
            self.clip_encoder = self.clip_encoder.to(device, dtype=dtype)
        return self

    def freeze(self) -> None:
        """Freeze text encoder parameters."""
        if self.t5_encoder is not None:
            for param in self.t5_encoder.parameters():
                param.requires_grad = False
        if self.clip_encoder is not None:
            for param in self.clip_encoder.parameters():
                param.requires_grad = False

    def parameters(self, recurse: bool = True):
        """Get all parameters."""
        params = []
        if self.t5_encoder is not None:
            params.extend(self.t5_encoder.parameters(recurse))
        if self.clip_encoder is not None:
            params.extend(self.clip_encoder.parameters(recurse))
        return iter(params)
