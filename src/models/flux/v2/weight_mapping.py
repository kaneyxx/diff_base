"""FLUX.2 weight key mapping between BFL native and internal (HF-aligned) formats.

This is the v2 counterpart of src/models/flux/v1/weight_mapping.py.

BFL native checkpoint keys use a different naming convention than the internal
HuggingFace-aligned naming used by Flux2Transformer.

Supported source formats:
- "bfl": BFL single-file safetensors (double_blocks / single_blocks naming)
- "hf": HuggingFace diffusers directory layout (transformer_blocks naming)
- "internal": Our internal naming — identical to HF naming, no conversion needed

Key differences from FLUX.1 mapping:
- Shared modulation: double_stream_modulation_img.*, double_stream_modulation_txt.*,
  single_stream_modulation.* (top-level, not per-block)
- Time/guidance embedding: time_guidance_embed.* (not time_text_embed.*)
- Pooled text: pooled_text_embed.* (separate from context_embedder)
- SwiGLU FF: ff.linear_in / ff.linear_out (not ff.net.0.proj / ff.net.2)
- Single blocks use fused to_qkv_mlp_proj (not split to_q/to_k/to_v + proj_mlp)
- All bias=False
- Double blocks have norm2 / norm2_context (no learnable params, no keys)
- No norm_out in FLUX.2 double blocks (norm_out is top-level only)

BFL key naming for FLUX.2:
  img_in.weight                           -> x_embedder.weight
  txt_in.weight                           -> context_embedder.weight
  vector_in.weight                        -> pooled_text_embed.weight
  time_in.in_proj.weight                  -> time_guidance_embed.timestep_embedder.linear_1.weight
  time_in.out_proj.weight                 -> time_guidance_embed.timestep_embedder.linear_2.weight
  guidance_in.in_proj.weight              -> time_guidance_embed.guidance_embedder.linear_1.weight
  guidance_in.out_proj.weight             -> time_guidance_embed.guidance_embedder.linear_2.weight
  img_mod.lin.weight                      -> double_stream_modulation_img.linear.weight
  txt_mod.lin.weight                      -> double_stream_modulation_txt.linear.weight
  single_mod.lin.weight                   -> single_stream_modulation.linear.weight
  final_layer.linear.weight               -> proj_out.weight
  final_layer.adaLN_modulation.1.weight   -> norm_out.linear.weight

  double_blocks.N.img_attn.to_q.weight    -> transformer_blocks.N.attn.to_q.weight
  double_blocks.N.img_attn.to_k.weight    -> transformer_blocks.N.attn.to_k.weight
  double_blocks.N.img_attn.to_v.weight    -> transformer_blocks.N.attn.to_v.weight
  double_blocks.N.img_attn.norm_q.weight  -> transformer_blocks.N.attn.norm_q.weight
  double_blocks.N.img_attn.norm_k.weight  -> transformer_blocks.N.attn.norm_k.weight
  double_blocks.N.img_attn.out.weight     -> transformer_blocks.N.attn.to_out.0.weight
  double_blocks.N.txt_attn.to_q.weight    -> transformer_blocks.N.attn.add_q_proj.weight
  double_blocks.N.txt_attn.to_k.weight    -> transformer_blocks.N.attn.add_k_proj.weight
  double_blocks.N.txt_attn.to_v.weight    -> transformer_blocks.N.attn.add_v_proj.weight
  double_blocks.N.txt_attn.norm_q.weight  -> transformer_blocks.N.attn.norm_added_q.weight
  double_blocks.N.txt_attn.norm_k.weight  -> transformer_blocks.N.attn.norm_added_k.weight
  double_blocks.N.txt_attn.out.weight     -> transformer_blocks.N.attn.to_add_out.weight
  double_blocks.N.img_mlp.linear_in.weight-> transformer_blocks.N.ff.linear_in.weight
  double_blocks.N.img_mlp.linear_out.weight-> transformer_blocks.N.ff.linear_out.weight
  double_blocks.N.txt_mlp.linear_in.weight-> transformer_blocks.N.ff_context.linear_in.weight
  double_blocks.N.txt_mlp.linear_out.weight-> transformer_blocks.N.ff_context.linear_out.weight

  single_blocks.N.to_qkv_mlp.weight      -> single_transformer_blocks.N.attn.to_qkv_mlp_proj.weight
  single_blocks.N.attn.norm_q.weight      -> single_transformer_blocks.N.attn.norm_q.weight
  single_blocks.N.attn.norm_k.weight      -> single_transformer_blocks.N.attn.norm_k.weight
  single_blocks.N.out.weight              -> single_transformer_blocks.N.attn.to_out.weight

Note: FLUX.2 internal format already stores the fused to_qkv_mlp_proj in single
blocks. BFL may also store it fused. No split/merge needed for single blocks.
"""

import re
from enum import Enum
from pathlib import Path
from typing import Literal

import torch

from ....utils.logging import get_logger

logger = get_logger(__name__)

# Type alias
StateDict = dict[str, torch.Tensor]

# Format enum (same values as v1 but re-declared for v2 independence)
SourceFormat = Literal["bfl", "hf", "internal"]


class Flux2FormatEnum(Enum):
    """Enumeration of recognised FLUX.2 checkpoint formats."""

    BFL = "bfl"
    HF = "hf"
    INTERNAL = "internal"


# ---------------------------------------------------------------------------
# Top-level (non-block) BFL -> internal static map
# ---------------------------------------------------------------------------

STATIC_MAP: dict[str, str] = {
    # Input projections (all bias=False in FLUX.2)
    "img_in.weight": "x_embedder.weight",
    "txt_in.weight": "context_embedder.weight",
    "vector_in.weight": "pooled_text_embed.weight",
    # Time / guidance embedding
    "time_in.in_proj.weight": "time_guidance_embed.timestep_embedder.linear_1.weight",
    "time_in.out_proj.weight": "time_guidance_embed.timestep_embedder.linear_2.weight",
    "guidance_in.in_proj.weight": "time_guidance_embed.guidance_embedder.linear_1.weight",
    "guidance_in.out_proj.weight": "time_guidance_embed.guidance_embedder.linear_2.weight",
    # Shared modulation (no per-block modulation in FLUX.2)
    "img_mod.lin.weight": "double_stream_modulation_img.linear.weight",
    "txt_mod.lin.weight": "double_stream_modulation_txt.linear.weight",
    "single_mod.lin.weight": "single_stream_modulation.linear.weight",
    # Output
    "final_layer.linear.weight": "proj_out.weight",
    "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
    "final_layer.adaLN_modulation.1.bias": "norm_out.linear.bias",
    # Fill mode embedder (optional — only present if model uses fill)
    "img_in_fill.weight": "x_embedder_fill.weight",
}

# Inverse map (internal -> BFL)
_STATIC_MAP_INV: dict[str, str] = {v: k for k, v in STATIC_MAP.items()}

# ---------------------------------------------------------------------------
# Double-block suffix maps
# ---------------------------------------------------------------------------

# BFL double-block suffix -> internal suffix (within transformer_blocks.N)
DOUBLE_BLOCK_SUFFIX_MAP: dict[str, str] = {
    # Image stream attention
    "img_attn.to_q.weight": "attn.to_q.weight",
    "img_attn.to_k.weight": "attn.to_k.weight",
    "img_attn.to_v.weight": "attn.to_v.weight",
    "img_attn.norm_q.weight": "attn.norm_q.weight",
    "img_attn.norm_k.weight": "attn.norm_k.weight",
    "img_attn.out.weight": "attn.to_out.0.weight",
    # Text stream attention (added_kv_proj naming)
    "txt_attn.to_q.weight": "attn.add_q_proj.weight",
    "txt_attn.to_k.weight": "attn.add_k_proj.weight",
    "txt_attn.to_v.weight": "attn.add_v_proj.weight",
    "txt_attn.norm_q.weight": "attn.norm_added_q.weight",
    "txt_attn.norm_k.weight": "attn.norm_added_k.weight",
    "txt_attn.out.weight": "attn.to_add_out.weight",
    # SwiGLU feed-forward (image stream)
    "img_mlp.linear_in.weight": "ff.linear_in.weight",
    "img_mlp.linear_out.weight": "ff.linear_out.weight",
    # SwiGLU feed-forward (text stream)
    "txt_mlp.linear_in.weight": "ff_context.linear_in.weight",
    "txt_mlp.linear_out.weight": "ff_context.linear_out.weight",
}

# Inverse
_DOUBLE_BLOCK_SUFFIX_MAP_INV: dict[str, str] = {v: k for k, v in DOUBLE_BLOCK_SUFFIX_MAP.items()}

# ---------------------------------------------------------------------------
# Single-block suffix maps
# ---------------------------------------------------------------------------

# BFL single-block suffix -> internal suffix (within single_transformer_blocks.N)
# FLUX.2 stores to_qkv_mlp_proj fused in both BFL and internal formats.
SINGLE_BLOCK_SUFFIX_MAP: dict[str, str] = {
    "to_qkv_mlp.weight": "attn.to_qkv_mlp_proj.weight",
    "attn.norm_q.weight": "attn.norm_q.weight",
    "attn.norm_k.weight": "attn.norm_k.weight",
    "out.weight": "attn.to_out.weight",
}

# Inverse
_SINGLE_BLOCK_SUFFIX_MAP_INV: dict[str, str] = {v: k for k, v in SINGLE_BLOCK_SUFFIX_MAP.items()}


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def detect_format(state_dict: StateDict) -> SourceFormat:
    """Detect the source format of a FLUX.2 state dict by inspecting key prefixes.

    Args:
        state_dict: Raw state dict loaded from checkpoint.

    Returns:
        "bfl" if keys use BFL native naming (double_blocks/img_in/etc.),
        "hf" or "internal" if keys use HF-aligned naming (transformer_blocks/x_embedder).
        "internal" and "hf" are the same naming convention for FLUX.2.

    Raises:
        ValueError: If format cannot be determined from the keys.
    """
    sample_keys = list(state_dict.keys())[:20]

    has_bfl = any(
        k.startswith((
            "double_blocks.", "single_blocks.", "img_in.", "txt_in.",
            "final_layer.", "img_mod.", "txt_mod.", "single_mod.",
        ))
        for k in sample_keys
    )
    has_internal = any(
        k.startswith((
            "transformer_blocks.", "single_transformer_blocks.",
            "x_embedder.", "context_embedder.", "pooled_text_embed.",
            "time_guidance_embed.", "double_stream_modulation_",
            "single_stream_modulation.",
        ))
        for k in sample_keys
    )

    if has_bfl and not has_internal:
        return "bfl"
    if has_internal and not has_bfl:
        return "internal"
    if not has_bfl and not has_internal:
        raise ValueError(
            "Cannot detect FLUX.2 checkpoint format: no recognized key prefixes found. "
            f"Sample keys: {sample_keys[:5]}"
        )
    raise ValueError(
        "Ambiguous FLUX.2 checkpoint format: found both BFL and internal-style keys. "
        f"Sample keys: {sample_keys[:5]}"
    )


# ---------------------------------------------------------------------------
# Conversion: BFL -> internal
# ---------------------------------------------------------------------------


def _convert_bfl_to_internal(state_dict: StateDict) -> StateDict:
    """Convert BFL FLUX.2 state dict to internal (HF-aligned) naming.

    Args:
        state_dict: BFL state dict.

    Returns:
        State dict with internal key naming.

    Raises:
        KeyError: If an unrecognized BFL key is encountered.
    """
    out: StateDict = {}
    unhandled: list[str] = []

    for bfl_key, val in state_dict.items():
        # Static top-level keys
        if bfl_key in STATIC_MAP:
            out[STATIC_MAP[bfl_key]] = val
            continue

        # Double blocks: double_blocks.N.<suffix>
        m = re.match(r"^double_blocks\.(\d+)\.", bfl_key)
        if m:
            n = m.group(1)
            suffix = bfl_key[m.end():]
            internal_prefix = f"transformer_blocks.{n}"

            if suffix in DOUBLE_BLOCK_SUFFIX_MAP:
                out[f"{internal_prefix}.{DOUBLE_BLOCK_SUFFIX_MAP[suffix]}"] = val
            else:
                unhandled.append(bfl_key)
            continue

        # Single blocks: single_blocks.N.<suffix>
        m = re.match(r"^single_blocks\.(\d+)\.", bfl_key)
        if m:
            n = m.group(1)
            suffix = bfl_key[m.end():]
            internal_prefix = f"single_transformer_blocks.{n}"

            if suffix in SINGLE_BLOCK_SUFFIX_MAP:
                out[f"{internal_prefix}.{SINGLE_BLOCK_SUFFIX_MAP[suffix]}"] = val
            else:
                unhandled.append(bfl_key)
            continue

        unhandled.append(bfl_key)

    if unhandled:
        raise KeyError(
            f"FLUX.2 BFL->internal conversion: {len(unhandled)} key(s) could not be mapped. "
            f"Unrecognized keys: {unhandled[:10]}"
            + (" (truncated)" if len(unhandled) > 10 else "")
        )

    return out


# ---------------------------------------------------------------------------
# Conversion: internal -> BFL
# ---------------------------------------------------------------------------


def _convert_internal_to_bfl(state_dict: StateDict) -> StateDict:
    """Convert internal FLUX.2 state dict to BFL native naming.

    This is the inverse of _convert_bfl_to_internal.

    Args:
        state_dict: State dict with internal (HF-aligned) key naming.

    Returns:
        State dict with BFL native key naming.
    """
    out: StateDict = {}

    for key, val in state_dict.items():
        # Static top-level keys
        if key in _STATIC_MAP_INV:
            out[_STATIC_MAP_INV[key]] = val
            continue

        # Double blocks: transformer_blocks.N.<suffix>
        m = re.match(r"^transformer_blocks\.(\d+)\.", key)
        if m:
            n = m.group(1)
            suffix = key[m.end():]
            bfl_prefix = f"double_blocks.{n}"

            if suffix in _DOUBLE_BLOCK_SUFFIX_MAP_INV:
                out[f"{bfl_prefix}.{_DOUBLE_BLOCK_SUFFIX_MAP_INV[suffix]}"] = val
            else:
                logger.warning(f"Unrecognized double block suffix: {suffix!r} (key={key!r})")
            continue

        # Single blocks: single_transformer_blocks.N.<suffix>
        m = re.match(r"^single_transformer_blocks\.(\d+)\.", key)
        if m:
            n = m.group(1)
            suffix = key[m.end():]
            bfl_prefix = f"single_blocks.{n}"

            if suffix in _SINGLE_BLOCK_SUFFIX_MAP_INV:
                out[f"{bfl_prefix}.{_SINGLE_BLOCK_SUFFIX_MAP_INV[suffix]}"] = val
            else:
                logger.warning(f"Unrecognized single block suffix: {suffix!r} (key={key!r})")
            continue

        logger.warning(f"Unrecognized internal key during BFL export: {key!r}")

    logger.info(f"Converted {len(state_dict)} internal keys to {len(out)} BFL keys")
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_state_dict(
    state_dict: StateDict,
    target: str = "internal",
    source: SourceFormat | None = None,
) -> StateDict:
    """Convert a FLUX.2 state dict between formats.

    Args:
        state_dict: Raw state dict to convert.
        target: Target format — "internal", "hf" (synonym for internal), or "bfl".
        source: Source format override. If None, auto-detected via detect_format().

    Returns:
        Converted state dict.

    Raises:
        ValueError: If target or detected source format is not recognised.
        KeyError: If a BFL key cannot be mapped (strict: unmapped keys raise).
    """
    if source is None:
        source = detect_format(state_dict)

    # Normalise target
    if target in ("internal", "hf"):
        target_norm = "internal"
    elif target == "bfl":
        target_norm = "bfl"
    else:
        raise ValueError(
            f"Unknown target format: {target!r}. Supported: 'internal', 'hf', 'bfl'."
        )

    # If already at target, no-op copy
    if source in ("internal", "hf") and target_norm == "internal":
        return dict(state_dict)
    if source == "bfl" and target_norm == "bfl":
        return dict(state_dict)

    # Conversions
    if source == "bfl" and target_norm == "internal":
        return _convert_bfl_to_internal(state_dict)

    if source in ("internal", "hf") and target_norm == "bfl":
        return _convert_internal_to_bfl(state_dict)

    raise ValueError(
        f"Unsupported conversion: source={source!r} -> target={target!r}."
    )


def load_flux2_checkpoint(
    path: str | Path,
    target: str = "internal",
) -> StateDict:
    """Load a FLUX.2 checkpoint and convert to the requested format.

    Auto-detects the source format (BFL native or HF/internal).

    Args:
        path: Path to a .safetensors or .bin/.pt/.pth file.
        target: Target format — "internal" (default), "hf", or "bfl".

    Returns:
        State dict in the requested naming convention.

    Raises:
        ValueError: If file format or checkpoint format is not recognised.
        KeyError: If a BFL key cannot be mapped during conversion.
    """
    from safetensors.torch import load_file

    p = Path(path)
    if p.suffix == ".safetensors":
        raw = load_file(str(p))
    elif p.suffix in (".pt", ".bin", ".pth"):
        raw = torch.load(str(p), map_location="cpu", weights_only=True)
    else:
        raise ValueError(
            f"Unsupported checkpoint extension: {p.suffix}. "
            "Supported: .safetensors, .bin, .pt, .pth"
        )

    fmt = detect_format(raw)
    logger.info(f"Detected FLUX.2 checkpoint format: '{fmt}' from {p.name}")

    return convert_state_dict(raw, target=target, source=fmt)
