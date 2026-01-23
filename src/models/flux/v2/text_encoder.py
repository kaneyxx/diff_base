"""FLUX.2 Text Encoders (Mistral and Qwen).

FLUX.2 variants use different text encoders than FLUX.1:
- dev: Mistral-3 (mistralai/Mistral-Small-3.1-24B-Instruct-2503)
- klein-4b: Qwen3-4B
- klein-9b: Qwen3-8B

These are LLM-based text encoders rather than CLIP/T5 combinations.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig


class Flux2TextEncoders(nn.Module):
    """FLUX.2 text encoders using Mistral or Qwen models.

    Uses HuggingFace transformers for the actual encoder models.
    """

    # Variant-specific encoder configurations
    ENCODER_CONFIGS = {
        "dev": {
            "model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            "encoder_type": "mistral",
            "hidden_size": 4096,
            "max_length": 512,
        },
        "klein-4b": {
            "model_id": "Qwen/Qwen3-4B",
            "encoder_type": "qwen",
            "hidden_size": 4096,
            "max_length": 512,
        },
        "klein-9b": {
            "model_id": "Qwen/Qwen3-8B",
            "encoder_type": "qwen",
            "hidden_size": 4096,
            "max_length": 512,
        },
    }

    def __init__(self, config: DictConfig, variant: str = "dev"):
        """Initialize FLUX.2 text encoder.

        Args:
            config: Text encoder configuration.
            variant: Model variant ("dev", "klein-4b", or "klein-9b").
        """
        super().__init__()
        self.config = config
        self.variant = variant

        # Get variant defaults
        encoder_cfg = self.ENCODER_CONFIGS.get(variant, self.ENCODER_CONFIGS["dev"])

        self.model_id = config.get("model_id", encoder_cfg["model_id"])
        self.encoder_type = config.get("type", encoder_cfg["encoder_type"])
        self.hidden_size = config.get("hidden_size", encoder_cfg["hidden_size"])
        self.max_length = config.get("max_length", encoder_cfg["max_length"])

        # Will be loaded from pretrained
        self.encoder = None
        self.tokenizer = None
        self._loaded = False

    def load_pretrained(self, pretrained_path: Optional[str | Path] = None) -> None:
        """Load pretrained text encoder.

        Args:
            pretrained_path: Path to model directory. If None, loads from HuggingFace.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = pretrained_path if pretrained_path else self.model_id

        if self.encoder_type == "mistral":
            self._load_mistral_encoder(model_path)
        elif self.encoder_type == "qwen":
            self._load_qwen_encoder(model_path)
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")

        self._loaded = True

    def _load_mistral_encoder(self, model_path: str | Path) -> None:
        """Load Mistral encoder.

        Args:
            model_path: Path to model or HuggingFace model ID.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        # Load as encoder-only (we only need hidden states, not generation)
        self.encoder = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            output_hidden_states=True,
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_qwen_encoder(self, model_path: str | Path) -> None:
        """Load Qwen encoder.

        Args:
            model_path: Path to model or HuggingFace model ID.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.encoder = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            output_hidden_states=True,
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(
        self,
        prompt: str | list[str],
        device: torch.device | str = "cuda",
        max_length: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Encode text prompts.

        Args:
            prompt: Text prompt or list of prompts.
            device: Target device.
            max_length: Maximum sequence length.

        Returns:
            Dictionary with:
                - prompt_embeds: Text hidden states
                - pooled_prompt_embeds: Pooled (last token) embeddings
                - attention_mask: Attention mask
        """
        if not self._loaded:
            raise RuntimeError("Text encoder not loaded. Call load_pretrained first.")

        if isinstance(prompt, str):
            prompt = [prompt]

        max_length = max_length or self.max_length

        # Tokenize
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)

        with torch.no_grad():
            outputs = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            # Get last hidden state
            hidden_states = outputs.hidden_states[-1]

            # Pool using last non-padding token
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(len(prompt), device=device)
            pooled_embeds = hidden_states[batch_indices, sequence_lengths]

        return {
            "prompt_embeds": hidden_states,
            "pooled_prompt_embeds": pooled_embeds,
            "attention_mask": attention_mask,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with pre-tokenized inputs.

        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask.

        Returns:
            Tuple of (hidden_states, pooled_embeds).
        """
        if not self._loaded:
            raise RuntimeError("Text encoder not loaded.")

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[-1]

        # Pool using last non-padding token
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
            pooled_embeds = hidden_states[batch_indices, sequence_lengths]
        else:
            pooled_embeds = hidden_states[:, -1]

        return hidden_states, pooled_embeds

    def to(self, device: torch.device | str, dtype: Optional[torch.dtype] = None):
        """Move encoder to device."""
        if self.encoder is not None:
            self.encoder = self.encoder.to(device, dtype=dtype)
        return self

    def freeze(self) -> None:
        """Freeze encoder parameters."""
        if self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def parameters(self, recurse: bool = True):
        """Get all parameters."""
        if self.encoder is not None:
            return self.encoder.parameters(recurse)
        return iter([])
