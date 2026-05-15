"""FLUX.1 Text Encoders (T5-XXL and CLIP-L).

FLUX.1 uses:
- T5-XXL (google/t5-v1_1-xxl) for main text embeddings
- CLIP-L (openai/clip-vit-large-patch14) for pooled embeddings
"""

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig


class Flux1TextEncoders(nn.Module):
    """FLUX.1 dual text encoders (T5-XXL and CLIP-L).

    Uses HuggingFace transformers for the actual encoder models.
    """

    def __init__(self, config: DictConfig):
        """Initialize FLUX.1 text encoders.

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
        # When True, encoder weights stay on CPU; encode() returns outputs on the
        # caller's requested device. Frees ~10 GB of GPU at the cost of CPU-side
        # token encoding (cheap since text encoders run once per training step).
        self.cpu_offload: bool = bool(config.get("cpu_offload", False))

    def load_pretrained(self, pretrained_path: str | Path) -> None:
        """Load pretrained text encoders.

        Args:
            pretrained_path: Path to Flux model directory.
        """
        from transformers import (
            CLIPTextModel,
            CLIPTokenizer,
            T5EncoderModel,
            T5Tokenizer,
        )

        pretrained_path = Path(pretrained_path)

        # Honour the dtype on the wrapper (set by Flux1Model from config.dtype)
        # to avoid the float32 → cast round-trip that doubles host RAM on T5.
        load_dtype = getattr(self, "dtype", torch.bfloat16)
        load_kwargs = {"torch_dtype": load_dtype, "low_cpu_mem_usage": True}

        # HF FLUX layout: text_encoder/ = CLIP-L, text_encoder_2/ = T5-XXL;
        # tokenizer/ and tokenizer_2/ live in matching slots.
        t5_enc_path = pretrained_path / "text_encoder_2"
        t5_tok_path = pretrained_path / "tokenizer_2"
        if t5_enc_path.exists() and t5_tok_path.exists():
            self.t5_encoder = T5EncoderModel.from_pretrained(t5_enc_path, **load_kwargs)
            self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_tok_path)
        else:
            self.t5_encoder = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl", **load_kwargs)
            self.t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")

        clip_enc_path = pretrained_path / "text_encoder"
        clip_tok_path = pretrained_path / "tokenizer"
        if clip_enc_path.exists() and clip_tok_path.exists():
            self.clip_encoder = CLIPTextModel.from_pretrained(clip_enc_path, **load_kwargs)
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_tok_path)
        else:
            self.clip_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14", **load_kwargs
            )
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )

        self._loaded = True

    def encode(
        self,
        prompt: str | list[str],
        device: torch.device | str = "cuda",
        max_t5_length: int | None = None,
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
                - attention_mask: T5 attention mask
        """
        if not self._loaded:
            raise RuntimeError("Text encoders not loaded. Call load_pretrained first.")

        if isinstance(prompt, str):
            prompt = [prompt]

        max_t5_length = max_t5_length or self.max_t5_length

        # When offloaded, run encoder forward where its weights live (CPU) and
        # only move outputs to the caller's device. Otherwise feed tokens to the
        # same device as the encoder weights.
        t5_device = next(self.t5_encoder.parameters()).device
        clip_device = next(self.clip_encoder.parameters()).device

        t5_tokens = self.t5_tokenizer(
            prompt, padding="max_length", max_length=max_t5_length,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            t5_output = self.t5_encoder(
                t5_tokens.input_ids.to(t5_device),
                attention_mask=t5_tokens.attention_mask.to(t5_device),
            )
            t5_embeds = t5_output.last_hidden_state.to(device)

        clip_tokens = self.clip_tokenizer(
            prompt, padding="max_length", max_length=self.max_clip_length,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            clip_output = self.clip_encoder(
                clip_tokens.input_ids.to(clip_device),
                attention_mask=clip_tokens.attention_mask.to(clip_device),
            )
            pooled_embeds = clip_output.pooler_output.to(device)

        return {
            "prompt_embeds": t5_embeds,
            "pooled_prompt_embeds": pooled_embeds,
            "attention_mask": t5_tokens.attention_mask.to(device),
        }

    def forward(
        self,
        t5_input_ids: torch.Tensor,
        clip_input_ids: torch.Tensor,
        t5_attention_mask: torch.Tensor | None = None,
        clip_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None):
        """Move encoders to device. Skips when ``cpu_offload`` is set."""
        if self.cpu_offload:
            # Honour dtype-only changes (e.g. bf16 cast) but keep weights on CPU.
            if dtype is not None:
                if self.t5_encoder is not None:
                    self.t5_encoder = self.t5_encoder.to(dtype=dtype)
                if self.clip_encoder is not None:
                    self.clip_encoder = self.clip_encoder.to(dtype=dtype)
            return self
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
