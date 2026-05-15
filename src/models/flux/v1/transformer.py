"""FLUX.1 Transformer (DiT) architecture.

FLUX.1 variants (dev, schnell) use:
- 19 joint transformer blocks
- 38 single transformer blocks
- 3072 hidden size, 24 attention heads
- T5-XXL + CLIP-L text encoders
- 16 latent channels

Image Editing Support:
- Kontext Mode: Reference image editing via sequence-wise concatenation
  - img_cond_seq is concatenated along sequence dimension (dim=1)
  - 3D Position IDs [stream, h, w] distinguish target (stream=0) from reference (stream=1)
- Fill Mode: NOT supported in FLUX.1

HuggingFace Alignment:
This transformer uses naming conventions compatible with HuggingFace's
FluxTransformer2DModel for direct weight loading from pretrained checkpoints.

RoPE Implementation (matches HuggingFace exactly):
- Position IDs are 3D: [stream, h, w] with axes_dim=(16, 56, 56) = 128 total
- Text position IDs are all zeros [0, 0, 0] -> identity RoPE (no rotation)
- Single unified image_rotary_emb = pos_embed(cat(txt_ids, img_ids))
- Same rotary_emb passed to both joint and single blocks
- Joint blocks split the emb internally into txt and img portions
- Timestep is multiplied by 1000 inside the model (caller passes 0-1 range)

Key naming mappings:
- transformer_blocks (not joint_blocks)
- single_transformer_blocks (not single_blocks)
- time_text_embed.timestep_embedder/guidance_embedder/text_embedder
- Blocks use to_q/to_k/to_v (not combined to_qkv)
"""

import math
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ...components.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
)
from ..base_transformer import FluxTransformerBase
from ..components.attention import FluxJointTransformerBlock, FluxSingleTransformerBlock
from ..components.embeddings import (
    compute_rope_from_position_ids,
)
from ..components.layers import AdaLayerNormContinuous
from .conditioning import create_position_ids

# Default FLUX.1 RoPE axes dimensions matching HuggingFace
FLUX1_AXES_DIM = (16, 56, 56)


class Flux1Transformer(FluxTransformerBase):
    """FLUX.1 DiT (Diffusion Transformer) architecture.

    Configuration for FLUX.1 variants:
    - dev: 19 joint + 38 single blocks, guidance enabled
    - schnell: 19 joint + 38 single blocks, guidance disabled

    Uses HuggingFace-compatible naming for direct weight loading.
    """

    # Default configurations for FLUX.1 variants
    VARIANT_CONFIGS = {
        "dev": {
            "num_layers": 19,
            "num_single_layers": 38,
            "hidden_size": 3072,
            "num_attention_heads": 24,
            "in_channels": 64,
            "guidance_embeds": True,
        },
        "schnell": {
            "num_layers": 19,
            "num_single_layers": 38,
            "hidden_size": 3072,
            "num_attention_heads": 24,
            "in_channels": 64,
            "guidance_embeds": False,
        },
        "kontext": {
            "num_layers": 19,
            "num_single_layers": 38,
            "hidden_size": 3072,
            "num_attention_heads": 24,
            "in_channels": 64,
            "guidance_embeds": True,
        },
    }

    def __init__(self, config: DictConfig, variant: str = "dev"):
        """Initialize FLUX.1 transformer.

        Args:
            config: Transformer configuration.
            variant: Model variant ("dev", "schnell", or "kontext").
        """
        super().__init__()
        self.config = config
        self.variant = variant

        if variant not in self.VARIANT_CONFIGS:
            raise ValueError(
                f"Unknown FLUX.1 variant: {variant!r}. "
                f"Supported: {sorted(self.VARIANT_CONFIGS)}"
            )

        # Get variant defaults, allow config overrides
        variant_cfg = self.VARIANT_CONFIGS[variant]

        hidden_size = config.get("hidden_size", variant_cfg["hidden_size"])
        num_heads = config.get("num_attention_heads", variant_cfg["num_attention_heads"])
        num_layers = config.get("num_layers", variant_cfg["num_layers"])
        num_single_layers = config.get("num_single_layers", variant_cfg["num_single_layers"])
        in_channels = config.get("in_channels", variant_cfg["in_channels"])
        pooled_projection_dim = config.get("pooled_projection_dim", 768)
        guidance_embeds = config.get("guidance_embeds", variant_cfg["guidance_embeds"])

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.guidance_embeds = guidance_embeds

        # RoPE configuration matching HuggingFace
        self.axes_dim = tuple(config.get("axes_dims_rope", FLUX1_AXES_DIM))
        self.rope_theta = config.get("rope_theta", 10000.0)

        # Input projection for latents
        self.x_embedder = nn.Linear(in_channels, hidden_size)

        # Context embedder for text (T5 4096 -> hidden_size 3072)
        joint_attention_dim = config.get("joint_attention_dim", 4096)
        self.context_embedder = nn.Linear(joint_attention_dim, hidden_size)

        # Combined time/guidance/text embedding (HF naming: time_text_embed)
        self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(
            embedding_dim=hidden_size,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
        )

        # Joint attention blocks (HF naming: transformer_blocks)
        self.transformer_blocks = nn.ModuleList([
            FluxJointTransformerBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        ])

        # Single stream blocks (HF naming: single_transformer_blocks)
        self.single_transformer_blocks = nn.ModuleList([
            FluxSingleTransformerBlock(hidden_size, num_heads)
            for _ in range(num_single_layers)
        ])

        # Output projection (HF-aligned: AdaLayerNormContinuous produces norm_out.linear.*)
        self.norm_out = AdaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=hidden_size,
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
        )
        self.proj_out = nn.Linear(hidden_size, in_channels)

        self._gradient_checkpointing = False

        # nn.ModuleDict so downstream conditioning modules participate in
        # state_dict and .to(device) without any extra bookkeeping.
        self.conditioning_modules: nn.ModuleDict = nn.ModuleDict()

    def register_conditioning_module(self, name: str, module: nn.Module) -> None:
        """Register a downstream conditioning module on this transformer.

        Stored in ``self.conditioning_modules`` (an ``nn.ModuleDict``) so the
        module participates in ``state_dict()`` save/load and ``.to(device)``
        calls automatically.

        Args:
            name: Unique key for the module (must be a valid Python identifier).
            module: The ``nn.Module`` to register.

        Raises:
            TypeError: If *module* is not an ``nn.Module``.
            ValueError: If *name* is already registered.
        """
        if not isinstance(module, nn.Module):
            raise TypeError(
                f"register_conditioning_module expects an nn.Module, got {type(module)}"
            )
        if name in self.conditioning_modules:
            raise ValueError(
                f"A conditioning module named '{name}' is already registered. "
                "Use a unique name or remove the existing one first."
            )
        self.conditioning_modules[name] = module

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor | None = None,
        img_ids: torch.Tensor | None = None,
        txt_ids: torch.Tensor | None = None,
        img_cond_seq: torch.Tensor | None = None,
        img_cond_seq_ids: torch.Tensor | None = None,
        return_hidden_states_at: list[int] | None = None,
        block_hooks: dict[Literal["joint", "single"], list[Callable]] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Forward pass matching HuggingFace FluxTransformer2DModel.

        RoPE flow (matches HuggingFace exactly):
        1. Create txt_ids = zeros(txt_seq, 3) and img_ids from spatial grid
        2. Concatenate: ids = cat(txt_ids, img_ids)  (along seq dim)
        3. Compute unified image_rotary_emb = pos_embed(ids)
        4. Pass same image_rotary_emb to both joint and single blocks
        5. Joint blocks split the emb into txt and img portions internally

        Timestep scaling:
        - Caller passes timestep in [0, 1] range
        - Model internally scales by 1000 (matching HuggingFace)

        Args:
            hidden_states: Latent tensor [B, seq_len, in_channels].
            timestep: Timestep values [B] in [0, 1] range.
            encoder_hidden_states: Text embeddings [B, txt_seq, joint_attention_dim].
            pooled_projections: Pooled text embeddings [B, pooled_dim].
            guidance: Optional guidance scale [B] in raw scale (e.g. 1.0).
            img_ids: Optional pre-computed image position IDs [B, img_seq, 3].
                If None, auto-computed from spatial dimensions.
            txt_ids: Optional pre-computed text position IDs [B, txt_seq, 3].
                If None, uses zeros (matching HuggingFace default).
            img_cond_seq: Kontext conditioning sequence [B, ref_seq, in_channels].
            img_cond_seq_ids: Position IDs for Kontext ref [B, ref_seq, 3].
            return_hidden_states_at: Optional list of joint block indices to capture.
            block_hooks: Optional dict with keys ``"joint"`` and/or ``"single"``,
                each mapping to a list of callables invoked after the corresponding
                block type. Each callable receives
                ``(block_idx, hidden_states, txt_hidden_or_None, temb)`` and must
                return a ``torch.Tensor`` delta that is **added** to
                ``hidden_states``. Defaults to ``None`` (no hooks, backward compat).

                **Experimental** — the hook signature may change between versions.

        Returns:
            Predicted noise [B, target_seq, in_channels] — target tokens only.
            Reference tokens (Kontext) are an internal implementation detail and
            are NOT included in the output.
            If return_hidden_states_at is set, returns (output, captured_states)
            where each captured state is also target-tokens-only.
        """
        B = hidden_states.shape[0]  # noqa: N806
        device = hidden_states.device
        dtype = hidden_states.dtype

        # === Timestep scaling (HuggingFace does this inside the model) ===
        timestep = timestep.to(dtype) * 1000
        if guidance is not None:
            if not self.guidance_embeds:
                raise ValueError(
                    f"FLUX.1 variant '{self.variant}' has guidance_embeds=False "
                    "(schnell). Do not pass guidance to schnell models."
                )
            guidance = guidance.to(dtype) * 1000

        # === Embed inputs ===
        hidden_states = self.x_embedder(hidden_states)

        # === Compute image position IDs if not provided ===
        base_seq_len = hidden_states.shape[1]
        if img_ids is None:
            h = w = int(math.sqrt(base_seq_len))
            if h * w != base_seq_len:
                h, w = 1, base_seq_len
            img_ids = create_position_ids(
                batch_size=B, height=h, width=w,
                device=device, dtype=dtype, time_offset=0.0,
            )

        # Capture target sequence length before Kontext concatenation so we
        # can slice reference tokens off the output later (AC1).
        target_seq_len = hidden_states.shape[1]

        # === Handle Kontext mode (sequence-wise concatenation) ===
        if img_cond_seq is not None:
            ref_embedded = self.x_embedder(img_cond_seq)
            hidden_states = torch.cat([hidden_states, ref_embedded], dim=1)

            if img_cond_seq_ids is not None:
                img_ids = torch.cat([img_ids, img_cond_seq_ids], dim=1)
            else:
                ref_seq_len = img_cond_seq.shape[1]
                ref_h = ref_w = int(math.sqrt(ref_seq_len))
                if ref_h * ref_w != ref_seq_len:
                    ref_h, ref_w = 1, ref_seq_len
                ref_ids = create_position_ids(
                    batch_size=B, height=ref_h, width=ref_w,
                    device=device, dtype=dtype, time_offset=1.0,
                )
                img_ids = torch.cat([img_ids, ref_ids], dim=1)

        # === Text position IDs (all zeros = identity RoPE) ===
        txt_seq_len = encoder_hidden_states.shape[1]
        if txt_ids is None:
            txt_ids = torch.zeros(B, txt_seq_len, 3, device=device, dtype=dtype)

        # === Unified RoPE: cat(txt_ids, img_ids) then compute ===
        # This matches HuggingFace: ids = cat(txt_ids, img_ids); emb = pos_embed(ids)
        combined_ids = torch.cat([txt_ids, img_ids], dim=1)
        image_rotary_emb = compute_rope_from_position_ids(
            combined_ids, sum(self.axes_dim), self.rope_theta,
            axes_dim=self.axes_dim,
        )

        # === Combined time/guidance/text embedding ===
        temb = self.time_text_embed(
            timestep=timestep,
            pooled_projection=pooled_projections,
            guidance=guidance,
        )

        # === Project text embeddings ===
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # === Split rotary emb for joint blocks ===
        # Joint blocks process txt and img separately, so split the unified emb
        txt_rotary_emb = (
            image_rotary_emb[0][:, :txt_seq_len],
            image_rotary_emb[1][:, :txt_seq_len],
        )
        img_rotary_emb = (
            image_rotary_emb[0][:, txt_seq_len:],
            image_rotary_emb[1][:, txt_seq_len:],
        )

        # === Joint attention blocks ===
        joint_hooks = block_hooks.get("joint", []) if block_hooks else []
        txt_hidden = encoder_hidden_states
        captured_hidden_states = {} if return_hidden_states_at is not None else None
        for block_idx, block in enumerate(self.transformer_blocks):
            hidden_states, txt_hidden = block(
                hidden_states,
                txt_hidden,
                temb,
                img_rotary_emb,
                txt_rotary_emb,
            )
            for hook in joint_hooks:
                delta = hook(block_idx, hidden_states, txt_hidden, temb)
                if not isinstance(delta, torch.Tensor):
                    raise TypeError(
                        f"block_hooks['joint'] hook at block {block_idx} must return "
                        f"a torch.Tensor delta, got {type(delta)}"
                    )
                hidden_states = hidden_states + delta
            if captured_hidden_states is not None and block_idx in return_hidden_states_at:
                # Capture target tokens only (REPA loss should not use reference tokens)
                captured_hidden_states[block_idx] = hidden_states[:, :target_seq_len]

        # === Concatenate for single stream ===
        hidden_states = torch.cat([txt_hidden, hidden_states], dim=1)

        # === Single stream blocks (use full unified rotary emb) ===
        single_hooks = block_hooks.get("single", []) if block_hooks else []
        for block_idx, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(hidden_states, temb, image_rotary_emb)
            for hook in single_hooks:
                delta = hook(block_idx, hidden_states, None, temb)
                if not isinstance(delta, torch.Tensor):
                    raise TypeError(
                        f"block_hooks['single'] hook at block {block_idx} must return "
                        f"a torch.Tensor delta, got {type(delta)}"
                    )
                hidden_states = hidden_states + delta

        # === Extract image tokens (remove text prefix, then remove reference tokens) ===
        hidden_states = hidden_states[:, txt_seq_len:]
        # Slice to target tokens only; in non-Kontext mode target_seq_len == full img seq.
        hidden_states = hidden_states[:, :target_seq_len]

        # === Project output ===
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if captured_hidden_states is not None:
            return output, captured_hidden_states
        return output

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing."""
        self._gradient_checkpointing = True
