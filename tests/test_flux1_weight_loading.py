"""Tests for FLUX.1 dual-format weight loading.

Covers:
- AC3: BFL native checkpoint loads with zero unexpected keys
- AC4: HF diffusers checkpoint loads with zero unexpected keys
- Format detection works correctly
- Converted weights match between formats
"""

import pytest
import torch
from omegaconf import OmegaConf

from src.models.flux.v1.transformer import Flux1Transformer
from src.models.flux.v1.weight_mapping import (
    convert_state_dict,
    detect_format,
)

# Tiny config matching the transformer we'll use in tests
TINY_CONFIG = OmegaConf.create({
    "hidden_size": 128,
    "num_attention_heads": 4,
    "num_layers": 1,
    "num_single_layers": 1,
    "in_channels": 64,
    "joint_attention_dim": 64,
    "pooled_projection_dim": 32,
    "axes_dims_rope": [8, 12, 12],
    "rope_theta": 10000.0,
})

HIDDEN = 128
HEADS = 4
HEAD_DIM = HIDDEN // HEADS  # 32
TXT_DIM = 64
POOL_DIM = 32
IN_CH = 64


def _make_internal_state_dict() -> dict:
    """Build a synthetic state dict in internal (HF-aligned) naming."""
    model = Flux1Transformer(TINY_CONFIG, variant="dev")
    return {k: v.clone() for k, v in model.state_dict().items()}


def _internal_to_bfl(internal: dict) -> dict:
    """Convert internal state dict to BFL naming (inverse of convert_state_dict).

    Used only for test fixture creation.
    """
    bfl = {}

    # Static reverse mappings
    REVERSE_STATIC = {  # noqa: N806
        "x_embedder.weight": "img_in.weight",
        "x_embedder.bias": "img_in.bias",
        "context_embedder.weight": "txt_in.weight",
        "context_embedder.bias": "txt_in.bias",
        "time_text_embed.timestep_embedder.linear_1.weight": "time_in.in_proj.weight",
        "time_text_embed.timestep_embedder.linear_1.bias": "time_in.in_proj.bias",
        "time_text_embed.timestep_embedder.linear_2.weight": "time_in.out_proj.weight",
        "time_text_embed.timestep_embedder.linear_2.bias": "time_in.out_proj.bias",
        "time_text_embed.text_embedder.linear_1.weight": "vector_in.in_proj.weight",
        "time_text_embed.text_embedder.linear_1.bias": "vector_in.in_proj.bias",
        "time_text_embed.text_embedder.linear_2.weight": "vector_in.out_proj.weight",
        "time_text_embed.text_embedder.linear_2.bias": "vector_in.out_proj.bias",
        "time_text_embed.guidance_embedder.linear_1.weight": "guidance_in.in_proj.weight",
        "time_text_embed.guidance_embedder.linear_1.bias": "guidance_in.in_proj.bias",
        "time_text_embed.guidance_embedder.linear_2.weight": "guidance_in.out_proj.weight",
        "time_text_embed.guidance_embedder.linear_2.bias": "guidance_in.out_proj.bias",
        "proj_out.weight": "final_layer.linear.weight",
        "proj_out.bias": "final_layer.linear.bias",
        "norm_out.linear.weight": "final_layer.adaLN_modulation.1.weight",
        "norm_out.linear.bias": "final_layer.adaLN_modulation.1.bias",
    }

    for n in range(1):  # 1 double block
        prefix_i = f"transformer_blocks.{n}"
        prefix_b = f"double_blocks.{n}"

        # Reverse single-key mappings for double blocks
        DOUBLE_REV = {  # noqa: N806
            f"{prefix_i}.attn.to_out.0.weight": f"{prefix_b}.img_attn.proj.weight",
            f"{prefix_i}.attn.to_out.0.bias": f"{prefix_b}.img_attn.proj.bias",
            f"{prefix_i}.attn.to_add_out.weight": f"{prefix_b}.txt_attn.proj.weight",
            f"{prefix_i}.attn.to_add_out.bias": f"{prefix_b}.txt_attn.proj.bias",
            f"{prefix_i}.attn.norm_q.weight": f"{prefix_b}.img_attn.norm.query_norm.scale",
            f"{prefix_i}.attn.norm_k.weight": f"{prefix_b}.img_attn.norm.key_norm.scale",
            f"{prefix_i}.attn.norm_added_q.weight": f"{prefix_b}.txt_attn.norm.query_norm.scale",
            f"{prefix_i}.attn.norm_added_k.weight": f"{prefix_b}.txt_attn.norm.key_norm.scale",
            f"{prefix_i}.ff.net.0.proj.weight": f"{prefix_b}.img_mlp.0.weight",
            f"{prefix_i}.ff.net.0.proj.bias": f"{prefix_b}.img_mlp.0.bias",
            f"{prefix_i}.ff.net.2.weight": f"{prefix_b}.img_mlp.2.weight",
            f"{prefix_i}.ff.net.2.bias": f"{prefix_b}.img_mlp.2.bias",
            f"{prefix_i}.ff_context.net.0.proj.weight": f"{prefix_b}.txt_mlp.0.weight",
            f"{prefix_i}.ff_context.net.0.proj.bias": f"{prefix_b}.txt_mlp.0.bias",
            f"{prefix_i}.ff_context.net.2.weight": f"{prefix_b}.txt_mlp.2.weight",
            f"{prefix_i}.ff_context.net.2.bias": f"{prefix_b}.txt_mlp.2.bias",
            f"{prefix_i}.norm1.linear.weight": f"{prefix_b}.img_mod.lin.weight",
            f"{prefix_i}.norm1.linear.bias": f"{prefix_b}.img_mod.lin.bias",
            f"{prefix_i}.norm1_context.linear.weight": f"{prefix_b}.txt_mod.lin.weight",
            f"{prefix_i}.norm1_context.linear.bias": f"{prefix_b}.txt_mod.lin.bias",
        }
        REVERSE_STATIC.update(DOUBLE_REV)

    for n in range(1):  # 1 single block
        prefix_i = f"single_transformer_blocks.{n}"
        prefix_b = f"single_blocks.{n}"

        SINGLE_REV = {  # noqa: N806
            f"{prefix_i}.proj_out.weight": f"{prefix_b}.linear2.weight",
            f"{prefix_i}.proj_out.bias": f"{prefix_b}.linear2.bias",
            f"{prefix_i}.norm.linear.weight": f"{prefix_b}.modulation.lin.weight",
            f"{prefix_i}.norm.linear.bias": f"{prefix_b}.modulation.lin.bias",
            f"{prefix_i}.attn.norm_q.weight": f"{prefix_b}.pre_norm.query_norm.scale",
            f"{prefix_i}.attn.norm_k.weight": f"{prefix_b}.pre_norm.key_norm.scale",
        }
        REVERSE_STATIC.update(SINGLE_REV)

    # Apply static reverse
    for k, v in internal.items():
        if k in REVERSE_STATIC:
            bfl[REVERSE_STATIC[k]] = v

    # Fuse qkv for double blocks
    for n in range(1):
        prefix_i = f"transformer_blocks.{n}"
        prefix_b = f"double_blocks.{n}"

        for suffix_pair in [("to_q", "to_k", "to_v", "img_attn.qkv"),
                             ("add_q_proj", "add_k_proj", "add_v_proj", "txt_attn.qkv")]:
            q_key, k_key, v_key, bfl_key = suffix_pair
            for wtype in ("weight", "bias"):
                q = internal.get(f"{prefix_i}.attn.{q_key}.{wtype}")
                k = internal.get(f"{prefix_i}.attn.{k_key}.{wtype}")
                v = internal.get(f"{prefix_i}.attn.{v_key}.{wtype}")
                if q is not None and k is not None and v is not None:
                    bfl[f"{prefix_b}.{bfl_key}.{wtype}"] = torch.cat([q, k, v], dim=0)

    # Fuse linear1 for single blocks
    for n in range(1):
        prefix_i = f"single_transformer_blocks.{n}"
        prefix_b = f"single_blocks.{n}"

        for wtype in ("weight", "bias"):
            q = internal.get(f"{prefix_i}.attn.to_q.{wtype}")
            k = internal.get(f"{prefix_i}.attn.to_k.{wtype}")
            v = internal.get(f"{prefix_i}.attn.to_v.{wtype}")
            mlp = internal.get(f"{prefix_i}.proj_mlp.{wtype}")
            if q is not None and k is not None and v is not None:
                parts = [q, k, v]
                if mlp is not None:
                    parts.append(mlp)
                bfl[f"{prefix_b}.linear1.{wtype}"] = torch.cat(parts, dim=0)

    return bfl


class TestFormatDetection:
    """Test detect_format() identifies BFL vs HF-aligned naming."""

    def test_detect_bfl_format(self):
        """State dict with 'double_blocks' keys → detected as 'bfl'."""
        sd = {
            "double_blocks.0.img_attn.qkv.weight": torch.zeros(3, 4),
            "img_in.weight": torch.zeros(4, 4),
        }
        assert detect_format(sd) == "bfl"

    def test_detect_diffusers_format(self):
        """State dict with 'transformer_blocks' keys → detected as 'diffusers'."""
        sd = {
            "transformer_blocks.0.attn.to_q.weight": torch.zeros(4, 4),
            "x_embedder.weight": torch.zeros(4, 4),
        }
        assert detect_format(sd) == "diffusers"

    def test_detect_internal_format(self):
        """Internal state dict (same as diffusers) detected as 'diffusers'."""
        internal = _make_internal_state_dict()
        fmt = detect_format(internal)
        assert fmt == "diffusers"

    def test_detect_unknown_raises(self):
        """Unrecognized key patterns raise ValueError."""
        sd = {"unknown.weight": torch.zeros(4, 4)}
        with pytest.raises(ValueError, match="Cannot detect checkpoint format"):
            detect_format(sd)


class TestBFLConversion:
    """Test BFL -> internal conversion (AC3)."""

    def test_bfl_roundtrip_loads_without_unexpected_keys(self):
        """BFL fixture state dict converts and loads with zero unexpected keys."""
        internal_ref = _make_internal_state_dict()
        bfl_sd = _internal_to_bfl(internal_ref)

        # Verify it looks like BFL format
        assert any(k.startswith("double_blocks.") for k in bfl_sd), \
            "Fixture should contain BFL-style double_blocks keys"

        # Convert back to internal
        converted = convert_state_dict(
            bfl_sd,
            source_format="bfl",
            num_heads=HEADS,
            hidden_size=HIDDEN,
        )

        # Load into model
        model = Flux1Transformer(TINY_CONFIG, variant="dev")
        missing, unexpected = model.load_state_dict(converted, strict=False)

        assert len(unexpected) == 0, (
            f"Unexpected keys after BFL conversion: {unexpected}"
        )

    def test_bfl_conversion_weights_match(self):
        """BFL conversion produces weights numerically identical to original."""
        internal_ref = _make_internal_state_dict()
        bfl_sd = _internal_to_bfl(internal_ref)
        converted = convert_state_dict(
            bfl_sd,
            source_format="bfl",
            num_heads=HEADS,
            hidden_size=HIDDEN,
        )

        # Compare a subset of keys
        for key in ["x_embedder.weight", "proj_out.weight", "transformer_blocks.0.attn.to_q.weight"]:
            if key in internal_ref and key in converted:
                diff = (internal_ref[key] - converted[key]).abs().max().item()
                assert diff < 1e-6, f"Weight mismatch for {key}: max_diff={diff}"

    def test_bfl_qkv_split_correct_shapes(self):
        """BFL qkv tensors split into correct per-head shapes."""
        # Create a known qkv tensor
        bfl_sd = {
            "double_blocks.0.img_attn.qkv.weight": torch.ones(HIDDEN * 3, HIDDEN),
            "double_blocks.0.img_attn.qkv.bias": torch.ones(HIDDEN * 3),
            # Minimal extra keys to avoid detection error
            "img_in.weight": torch.zeros(HIDDEN, IN_CH),
            "img_in.bias": torch.zeros(HIDDEN),
        }

        converted = convert_state_dict(
            bfl_sd,
            source_format="bfl",
            num_heads=HEADS,
            hidden_size=HIDDEN,
        )

        assert "transformer_blocks.0.attn.to_q.weight" in converted
        assert converted["transformer_blocks.0.attn.to_q.weight"].shape == (HIDDEN, HIDDEN)
        assert converted["transformer_blocks.0.attn.to_k.weight"].shape == (HIDDEN, HIDDEN)
        assert converted["transformer_blocks.0.attn.to_v.weight"].shape == (HIDDEN, HIDDEN)


class TestDiffusersConversion:
    """Test HF diffusers format loading (AC4)."""

    def test_diffusers_passthrough_no_unexpected_keys(self):
        """Internal/diffusers state dict loads with zero unexpected keys."""
        internal = _make_internal_state_dict()

        # convert_state_dict with diffusers format is a no-op
        converted = convert_state_dict(internal, source_format="diffusers")
        assert converted == internal  # passthrough

        model = Flux1Transformer(TINY_CONFIG, variant="dev")
        missing, unexpected = model.load_state_dict(converted, strict=False)

        assert len(unexpected) == 0, (
            f"Unexpected keys in diffusers passthrough: {unexpected}"
        )

    def test_internal_passthrough_no_unexpected_keys(self):
        """source_format='internal' is also a passthrough."""
        internal = _make_internal_state_dict()
        converted = convert_state_dict(internal, source_format="internal")
        assert converted == internal


class TestConversionErrors:
    """Test error handling in weight mapping."""

    def test_unrecognized_bfl_key_raises(self):
        """Unrecognized BFL key raises KeyError."""
        bad_sd = {
            "img_in.weight": torch.zeros(4, 4),
            "img_in.bias": torch.zeros(4),
            "some.totally.unknown.key": torch.zeros(4),
        }
        with pytest.raises(KeyError, match="could not be mapped"):
            convert_state_dict(bad_sd, source_format="bfl", num_heads=4, hidden_size=4)

    def test_unknown_format_raises(self):
        """Unknown source_format raises ValueError."""
        with pytest.raises(ValueError, match="Unknown source format"):
            convert_state_dict({}, source_format="unknown")  # type: ignore
