"""SDXL Text Encoders (CLIP-L and CLIP-G)."""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig


class SDXLTextEncoders(nn.Module):
    """SDXL dual text encoders (CLIP-L and CLIP-G with projection).

    Uses HuggingFace transformers for the actual CLIP models.
    """

    def __init__(self, config: DictConfig):
        """Initialize SDXL text encoders.

        Args:
            config: Text encoder configuration.
        """
        super().__init__()
        self.config = config

        # Will be loaded from pretrained
        self.text_encoder = None
        self.text_encoder_2 = None
        self.tokenizer = None
        self.tokenizer_2 = None

        self._loaded = False

    def load_pretrained(self, pretrained_path: str | Path) -> None:
        """Load pretrained text encoders from diffusers format.

        Args:
            pretrained_path: Path to SDXL model directory.
        """
        from transformers import (
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPTokenizer,
        )

        pretrained_path = Path(pretrained_path)

        # CLIP-L (text_encoder)
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_path / "text_encoder"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_path / "tokenizer"
        )

        # CLIP-G with projection (text_encoder_2)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            pretrained_path / "text_encoder_2"
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            pretrained_path / "tokenizer_2"
        )

        self._loaded = True

    def encode(
        self,
        prompt: str | list[str],
        device: torch.device | str = "cuda",
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str | list[str]] = None,
    ) -> dict[str, torch.Tensor]:
        """Encode text prompts.

        Args:
            prompt: Text prompt or list of prompts.
            device: Target device.
            num_images_per_prompt: Number of images per prompt.
            do_classifier_free_guidance: Whether to generate uncond embeddings.
            negative_prompt: Negative prompt for CFG.

        Returns:
            Dictionary with:
                - prompt_embeds: Combined text embeddings [B, 77, 2048]
                - pooled_prompt_embeds: Pooled embeddings [B, 1280]
        """
        if not self._loaded:
            raise RuntimeError("Text encoders not loaded. Call load_pretrained first.")

        if isinstance(prompt, str):
            prompt = [prompt]

        batch_size = len(prompt)

        # Tokenize for both encoders
        text_inputs_1 = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        # Encode with CLIP-L
        text_input_ids_1 = text_inputs_1.input_ids.to(device)
        prompt_embeds_1 = self.text_encoder(
            text_input_ids_1,
            output_hidden_states=True,
        )
        # Use penultimate hidden state
        prompt_embeds_1 = prompt_embeds_1.hidden_states[-2]

        # Encode with CLIP-G
        text_input_ids_2 = text_inputs_2.input_ids.to(device)
        prompt_embeds_2 = self.text_encoder_2(
            text_input_ids_2,
            output_hidden_states=True,
        )
        pooled_prompt_embeds = prompt_embeds_2.text_embeds
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]

        # Concatenate embeddings from both encoders
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

        # Handle classifier-free guidance
        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size

            uncond_embeds, uncond_pooled = self._encode_prompt(
                negative_prompt, device
            )

            # Duplicate for num_images_per_prompt
            prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1, 1)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(num_images_per_prompt, 1)
            uncond_embeds = uncond_embeds.repeat(num_images_per_prompt, 1, 1)
            uncond_pooled = uncond_pooled.repeat(num_images_per_prompt, 1)

            # Concat for CFG (uncond first, then cond)
            prompt_embeds = torch.cat([uncond_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([uncond_pooled, pooled_prompt_embeds], dim=0)

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }

    def _encode_prompt(
        self,
        prompt: list[str],
        device: torch.device | str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a single prompt without CFG handling."""
        # Tokenize
        text_inputs_1 = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        # Encode
        with torch.no_grad():
            text_input_ids_1 = text_inputs_1.input_ids.to(device)
            prompt_embeds_1 = self.text_encoder(
                text_input_ids_1,
                output_hidden_states=True,
            ).hidden_states[-2]

            text_input_ids_2 = text_inputs_2.input_ids.to(device)
            outputs_2 = self.text_encoder_2(
                text_input_ids_2,
                output_hidden_states=True,
            )
            pooled = outputs_2.text_embeds
            prompt_embeds_2 = outputs_2.hidden_states[-2]

        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

        return prompt_embeds, pooled

    def forward(
        self,
        input_ids_1: torch.Tensor,
        input_ids_2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with pre-tokenized inputs.

        Args:
            input_ids_1: Token IDs for CLIP-L [B, 77].
            input_ids_2: Token IDs for CLIP-G [B, 77].

        Returns:
            Tuple of (prompt_embeds, pooled_embeds).
        """
        if not self._loaded:
            raise RuntimeError("Text encoders not loaded.")

        # CLIP-L
        prompt_embeds_1 = self.text_encoder(
            input_ids_1,
            output_hidden_states=True,
        ).hidden_states[-2]

        # CLIP-G
        outputs_2 = self.text_encoder_2(
            input_ids_2,
            output_hidden_states=True,
        )
        pooled = outputs_2.text_embeds
        prompt_embeds_2 = outputs_2.hidden_states[-2]

        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

        return prompt_embeds, pooled

    def to(self, device: torch.device | str, dtype: Optional[torch.dtype] = None):
        """Move encoders to device."""
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(device, dtype=dtype)
        if self.text_encoder_2 is not None:
            self.text_encoder_2 = self.text_encoder_2.to(device, dtype=dtype)
        return self

    def train(self, mode: bool = True):
        """Set training mode."""
        if self.text_encoder is not None:
            self.text_encoder.train(mode)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def parameters(self, recurse: bool = True):
        """Get all parameters."""
        params = []
        if self.text_encoder is not None:
            params.extend(self.text_encoder.parameters(recurse))
        if self.text_encoder_2 is not None:
            params.extend(self.text_encoder_2.parameters(recurse))
        return iter(params)

    def freeze(self) -> None:
        """Freeze text encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze text encoder parameters."""
        for param in self.parameters():
            param.requires_grad = True
