"""FLUX.1 weight key mapping between BFL native and internal (HF-aligned) formats.

BFL native checkpoint keys (e.g., flux1-dev.safetensors from Black Forest Labs)
use a different naming convention than the internal HuggingFace-aligned naming.

Supported source formats:
- "bfl": BFL single-file safetensors (double_blocks / single_blocks naming)
- "diffusers": HuggingFace diffusers directory layout (transformer_blocks naming)
- "internal": Our internal naming — identical to diffusers naming, no conversion needed

BFL key naming:
  img_in.weight / img_in.bias               -> x_embedder.weight / .bias
  txt_in.weight / txt_in.bias               -> context_embedder.weight / .bias
  time_in.in_proj.weight / .bias            -> time_text_embed.timestep_embedder.linear_1.*
  time_in.out_proj.weight / .bias           -> time_text_embed.timestep_embedder.linear_2.*
  vector_in.in_proj.weight / .bias          -> time_text_embed.text_embedder.linear_1.*
  vector_in.out_proj.weight / .bias         -> time_text_embed.text_embedder.linear_2.*
  guidance_in.in_proj.weight / .bias        -> time_text_embed.guidance_embedder.linear_1.*
  guidance_in.out_proj.weight / .bias       -> time_text_embed.guidance_embedder.linear_2.*
  final_layer.linear.weight / .bias         -> proj_out.weight / .bias
  final_layer.adaLN_modulation.1.weight/.b  -> norm_out.linear.weight / .bias
  double_blocks.N.img_attn.qkv.weight       -> split -> transformer_blocks.N.attn.to_q/k/v.weight
  double_blocks.N.img_attn.proj.weight/.b   -> transformer_blocks.N.attn.to_out.0.weight / .bias
  double_blocks.N.txt_attn.qkv.weight       -> split -> transformer_blocks.N.attn.add_q/k/v_proj.weight
  double_blocks.N.txt_attn.proj.weight/.b   -> transformer_blocks.N.attn.to_add_out.weight / .bias
  double_blocks.N.img_attn.norm.query_norm.scale -> transformer_blocks.N.attn.norm_q.weight
  double_blocks.N.img_attn.norm.key_norm.scale   -> transformer_blocks.N.attn.norm_k.weight
  double_blocks.N.txt_attn.norm.query_norm.scale -> transformer_blocks.N.attn.norm_added_q.weight
  double_blocks.N.txt_attn.norm.key_norm.scale   -> transformer_blocks.N.attn.norm_added_k.weight
  double_blocks.N.img_mlp.0.weight / .bias  -> transformer_blocks.N.ff.net.0.proj.weight / .bias
  double_blocks.N.img_mlp.2.weight / .bias  -> transformer_blocks.N.ff.net.2.weight / .bias
  double_blocks.N.txt_mlp.0.weight / .bias  -> transformer_blocks.N.ff_context.net.0.proj.weight / .bias
  double_blocks.N.txt_mlp.2.weight / .bias  -> transformer_blocks.N.ff_context.net.2.weight / .bias
  double_blocks.N.img_mod.lin.weight / .b   -> transformer_blocks.N.norm1.linear.weight / .bias
  double_blocks.N.txt_mod.lin.weight / .b   -> transformer_blocks.N.norm1_context.linear.weight / .bias
  single_blocks.N.linear1.weight            -> split -> single_transformer_blocks.N.attn.to_q/k/v + proj_mlp (fused qkv_mlp)
  single_blocks.N.linear2.weight / .bias    -> single_transformer_blocks.N.proj_out.weight / .bias
  single_blocks.N.modulation.lin.weight/.b  -> single_transformer_blocks.N.norm.linear.weight / .bias
  single_blocks.N.pre_norm.query_norm.scale -> single_transformer_blocks.N.attn.norm_q.weight
  single_blocks.N.pre_norm.key_norm.scale   -> single_transformer_blocks.N.attn.norm_k.weight

Note: BFL fuses q/k/v into a single qkv tensor. Conversion splits along dim=0.
The split sizes are: [hidden_size, hidden_size, hidden_size] for img_attn,
and same for txt_attn. For single_blocks.linear1, it fuses [q, k, v, mlp_proj].
"""

import re
from typing import Literal

import torch

from ....utils.logging import get_logger

logger = get_logger(__name__)

# Type alias
StateDict = dict[str, torch.Tensor]
SourceFormat = Literal["bfl", "diffusers", "internal"]


def detect_format(state_dict: StateDict) -> SourceFormat:
    """Detect the source format of a FLUX.1 state dict by inspecting key prefixes.

    Args:
        state_dict: Raw state dict loaded from checkpoint.

    Returns:
        "bfl" if keys use BFL native naming (double_blocks/img_in/etc.),
        "diffusers" or "internal" if keys use HF-aligned naming (transformer_blocks/x_embedder).
        "internal" and "diffusers" are the same naming convention.

    Raises:
        ValueError: If format cannot be determined from the keys.
    """
    sample_keys = list(state_dict.keys())[:20]

    has_bfl = any(
        k.startswith(("double_blocks.", "single_blocks.", "img_in.", "txt_in.", "final_layer."))
        for k in sample_keys
    )
    has_hf = any(
        k.startswith(("transformer_blocks.", "single_transformer_blocks.", "x_embedder.", "context_embedder."))
        for k in sample_keys
    )

    if has_bfl and not has_hf:
        return "bfl"
    if has_hf and not has_bfl:
        # Both diffusers and internal use the same key naming
        return "diffusers"
    if not has_bfl and not has_hf:
        raise ValueError(
            "Cannot detect checkpoint format: no recognized key prefixes found. "
            f"Sample keys: {sample_keys[:5]}"
        )
    raise ValueError(
        "Ambiguous checkpoint format: found both BFL and HF-style keys. "
        f"Sample keys: {sample_keys[:5]}"
    )


def convert_state_dict(
    state_dict: StateDict,
    source_format: SourceFormat,
    num_heads: int = 24,
    hidden_size: int = 3072,
    num_double_blocks: int = 19,
    num_single_blocks: int = 38,
) -> StateDict:
    """Convert a FLUX.1 state dict from source_format to internal (HF-aligned) naming.

    Args:
        state_dict: Raw state dict in source_format.
        source_format: One of "bfl", "diffusers", "internal".
        num_heads: Number of attention heads (for qkv split). Default: 24.
        hidden_size: Transformer hidden size (for qkv split). Default: 3072.
        num_double_blocks: Number of double (joint) blocks. Default: 19.
        num_single_blocks: Number of single blocks. Default: 38.

    Returns:
        State dict with internal (HF-aligned) key naming.

    Raises:
        ValueError: If source_format is not recognized.
        KeyError: If a BFL key cannot be mapped (strict: unmapped keys raise).
    """
    if source_format in ("diffusers", "internal"):
        return dict(state_dict)  # already in internal format

    if source_format != "bfl":
        raise ValueError(
            f"Unknown source format: {source_format!r}. "
            "Supported: 'bfl', 'diffusers', 'internal'."
        )

    return _convert_bfl_to_internal(
        state_dict,
        num_heads=num_heads,
        hidden_size=hidden_size,
        num_double_blocks=num_double_blocks,
        num_single_blocks=num_single_blocks,
    )


def _split_qkv(
    qkv: torch.Tensor,
    num_heads: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a fused qkv weight tensor into q, k, v.

    Args:
        qkv: Fused tensor of shape [3*hidden_size, ...] or [3*hidden_size].
        num_heads: Number of attention heads (unused but kept for clarity).
        hidden_size: Size of each of q, k, v.

    Returns:
        Tuple (q, k, v), each of shape [hidden_size, ...].
    """
    return qkv.chunk(3, dim=0)


def _convert_bfl_to_internal(
    state_dict: StateDict,
    num_heads: int,
    hidden_size: int,
    num_double_blocks: int,
    num_single_blocks: int,
) -> StateDict:
    """Perform the actual BFL -> internal key conversion.

    Fused qkv tensors are split; all other tensors are renamed in-place (no copy
    of data unless splitting is required).

    Args:
        state_dict: BFL state dict.
        num_heads: Number of attention heads.
        hidden_size: Transformer hidden size.
        num_double_blocks: Number of double blocks.
        num_single_blocks: Number of single blocks.

    Returns:
        Converted state dict with internal naming.

    Raises:
        KeyError: If an unrecognized BFL key is encountered.
    """
    out: StateDict = {}
    unhandled: list[str] = []

    # --- Static top-level mappings (BFL key -> internal key) ---
    STATIC_MAP: dict[str, str] = {  # noqa: N806
        "img_in.weight": "x_embedder.weight",
        "img_in.bias": "x_embedder.bias",
        "txt_in.weight": "context_embedder.weight",
        "txt_in.bias": "context_embedder.bias",
        "time_in.in_proj.weight": "time_text_embed.timestep_embedder.linear_1.weight",
        "time_in.in_proj.bias": "time_text_embed.timestep_embedder.linear_1.bias",
        "time_in.out_proj.weight": "time_text_embed.timestep_embedder.linear_2.weight",
        "time_in.out_proj.bias": "time_text_embed.timestep_embedder.linear_2.bias",
        "vector_in.in_proj.weight": "time_text_embed.text_embedder.linear_1.weight",
        "vector_in.in_proj.bias": "time_text_embed.text_embedder.linear_1.bias",
        "vector_in.out_proj.weight": "time_text_embed.text_embedder.linear_2.weight",
        "vector_in.out_proj.bias": "time_text_embed.text_embedder.linear_2.bias",
        "guidance_in.in_proj.weight": "time_text_embed.guidance_embedder.linear_1.weight",
        "guidance_in.in_proj.bias": "time_text_embed.guidance_embedder.linear_1.bias",
        "guidance_in.out_proj.weight": "time_text_embed.guidance_embedder.linear_2.weight",
        "guidance_in.out_proj.bias": "time_text_embed.guidance_embedder.linear_2.bias",
        "final_layer.linear.weight": "proj_out.weight",
        "final_layer.linear.bias": "proj_out.bias",
        "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
        "final_layer.adaLN_modulation.1.bias": "norm_out.linear.bias",
    }

    for bfl_key, val in state_dict.items():
        if bfl_key in STATIC_MAP:
            out[STATIC_MAP[bfl_key]] = val
            continue

        # --- Double blocks ---
        m = re.match(r"^double_blocks\.(\d+)\.", bfl_key)
        if m:
            n = m.group(1)
            suffix = bfl_key[m.end():]
            internal_prefix = f"transformer_blocks.{n}"

            mapped = _map_double_block_key(suffix, internal_prefix, val, num_heads, hidden_size, out)
            if not mapped:
                unhandled.append(bfl_key)
            continue

        # --- Single blocks ---
        m = re.match(r"^single_blocks\.(\d+)\.", bfl_key)
        if m:
            n = m.group(1)
            suffix = bfl_key[m.end():]
            internal_prefix = f"single_transformer_blocks.{n}"

            mapped = _map_single_block_key(suffix, internal_prefix, val, num_heads, hidden_size, out)
            if not mapped:
                unhandled.append(bfl_key)
            continue

        unhandled.append(bfl_key)

    if unhandled:
        raise KeyError(
            f"BFL->internal conversion: {len(unhandled)} key(s) could not be mapped. "
            f"Unrecognized keys: {unhandled[:10]}"
            + (" (truncated)" if len(unhandled) > 10 else "")
        )

    return out


def _map_double_block_key(
    suffix: str,
    internal_prefix: str,
    val: torch.Tensor,
    num_heads: int,
    hidden_size: int,
    out: StateDict,
) -> bool:
    """Map a single double_blocks suffix to internal keys.

    Fused qkv tensors trigger split and produce 3 output keys.

    Args:
        suffix: Key suffix after "double_blocks.N." prefix.
        internal_prefix: Corresponding "transformer_blocks.N" prefix.
        val: Weight tensor.
        num_heads: Number of attention heads.
        hidden_size: Transformer hidden dimension.
        out: Output dict to write into.

    Returns:
        True if handled, False if unrecognized.
    """
    DOUBLE_SUFFIX_MAP: dict[str, str] = {  # noqa: N806
        "img_attn.proj.weight": "attn.to_out.0.weight",
        "img_attn.proj.bias": "attn.to_out.0.bias",
        "txt_attn.proj.weight": "attn.to_add_out.weight",
        "txt_attn.proj.bias": "attn.to_add_out.bias",
        "img_attn.norm.query_norm.scale": "attn.norm_q.weight",
        "img_attn.norm.key_norm.scale": "attn.norm_k.weight",
        "txt_attn.norm.query_norm.scale": "attn.norm_added_q.weight",
        "txt_attn.norm.key_norm.scale": "attn.norm_added_k.weight",
        "img_mlp.0.weight": "ff.net.0.proj.weight",
        "img_mlp.0.bias": "ff.net.0.proj.bias",
        "img_mlp.2.weight": "ff.net.2.weight",
        "img_mlp.2.bias": "ff.net.2.bias",
        "txt_mlp.0.weight": "ff_context.net.0.proj.weight",
        "txt_mlp.0.bias": "ff_context.net.0.proj.bias",
        "txt_mlp.2.weight": "ff_context.net.2.weight",
        "txt_mlp.2.bias": "ff_context.net.2.bias",
        "img_mod.lin.weight": "norm1.linear.weight",
        "img_mod.lin.bias": "norm1.linear.bias",
        "txt_mod.lin.weight": "norm1_context.linear.weight",
        "txt_mod.lin.bias": "norm1_context.linear.bias",
    }

    if suffix in DOUBLE_SUFFIX_MAP:
        out[f"{internal_prefix}.{DOUBLE_SUFFIX_MAP[suffix]}"] = val
        return True

    # Fused img qkv → split to q, k, v
    if suffix == "img_attn.qkv.weight":
        q, k, v = _split_qkv(val, num_heads, hidden_size)
        out[f"{internal_prefix}.attn.to_q.weight"] = q
        out[f"{internal_prefix}.attn.to_k.weight"] = k
        out[f"{internal_prefix}.attn.to_v.weight"] = v
        return True
    if suffix == "img_attn.qkv.bias":
        q, k, v = _split_qkv(val, num_heads, hidden_size)
        out[f"{internal_prefix}.attn.to_q.bias"] = q
        out[f"{internal_prefix}.attn.to_k.bias"] = k
        out[f"{internal_prefix}.attn.to_v.bias"] = v
        return True

    # Fused txt qkv → split to add_q_proj, add_k_proj, add_v_proj
    if suffix == "txt_attn.qkv.weight":
        q, k, v = _split_qkv(val, num_heads, hidden_size)
        out[f"{internal_prefix}.attn.add_q_proj.weight"] = q
        out[f"{internal_prefix}.attn.add_k_proj.weight"] = k
        out[f"{internal_prefix}.attn.add_v_proj.weight"] = v
        return True
    if suffix == "txt_attn.qkv.bias":
        q, k, v = _split_qkv(val, num_heads, hidden_size)
        out[f"{internal_prefix}.attn.add_q_proj.bias"] = q
        out[f"{internal_prefix}.attn.add_k_proj.bias"] = k
        out[f"{internal_prefix}.attn.add_v_proj.bias"] = v
        return True

    return False


def _map_single_block_key(
    suffix: str,
    internal_prefix: str,
    val: torch.Tensor,
    num_heads: int,
    hidden_size: int,
    out: StateDict,
) -> bool:
    """Map a single_blocks suffix to internal keys.

    single_blocks.N.linear1 is a fused [q, k, v, mlp_proj] weight.
    Split sizes: q=hidden_size, k=hidden_size, v=hidden_size, mlp=mlp_hidden.
    The mlp projection size is inferred from the tensor.

    Args:
        suffix: Key suffix after "single_blocks.N." prefix.
        internal_prefix: Corresponding "single_transformer_blocks.N" prefix.
        val: Weight tensor.
        num_heads: Number of attention heads.
        hidden_size: Transformer hidden dimension.
        out: Output dict to write into.

    Returns:
        True if handled, False if unrecognized.
    """
    SINGLE_SUFFIX_MAP: dict[str, str] = {  # noqa: N806
        "linear2.weight": "proj_out.weight",
        "linear2.bias": "proj_out.bias",
        "modulation.lin.weight": "norm.linear.weight",
        "modulation.lin.bias": "norm.linear.bias",
        "pre_norm.query_norm.scale": "attn.norm_q.weight",
        "pre_norm.key_norm.scale": "attn.norm_k.weight",
    }

    if suffix in SINGLE_SUFFIX_MAP:
        out[f"{internal_prefix}.{SINGLE_SUFFIX_MAP[suffix]}"] = val
        return True

    # Fused linear1: [q, k, v, mlp_proj] concatenated on dim=0
    if suffix in ("linear1.weight", "linear1.bias"):
        is_weight = suffix == "linear1.weight"
        # q, k, v each have size hidden_size; mlp gets the remainder
        qkv_size = hidden_size * 3
        mlp_size = val.shape[0] - qkv_size
        if mlp_size <= 0:
            # Fallback: treat as pure qkv (some older checkpoints)
            q, k, v = _split_qkv(val, num_heads, hidden_size)
        else:
            qkv, mlp = val[:qkv_size], val[qkv_size:]
            q, k, v = _split_qkv(qkv, num_heads, hidden_size)

        suffix_type = "weight" if is_weight else "bias"
        out[f"{internal_prefix}.attn.to_q.{suffix_type}"] = q
        out[f"{internal_prefix}.attn.to_k.{suffix_type}"] = k
        out[f"{internal_prefix}.attn.to_v.{suffix_type}"] = v
        if mlp_size > 0:
            out[f"{internal_prefix}.proj_mlp.{suffix_type}"] = mlp
        return True

    return False


def load_flux1_checkpoint(
    path: str,
    strict: bool = False,
    num_heads: int = 24,
    hidden_size: int = 3072,
    num_double_blocks: int = 19,
    num_single_blocks: int = 38,
) -> StateDict:
    """Load a FLUX.1 checkpoint and convert to internal naming.

    Auto-detects format (BFL native or HF diffusers/internal).
    Logs missing and unexpected keys at INFO level.

    Args:
        path: Path to a .safetensors or .bin file.
        strict: If True, raise on any unrecognized key after conversion.
            Default False for ergonomics; True recommended for production.
        num_heads: Number of attention heads for qkv split.
        hidden_size: Transformer hidden size for qkv split.
        num_double_blocks: Number of double blocks.
        num_single_blocks: Number of single blocks.

    Returns:
        State dict in internal (HF-aligned) naming.

    Raises:
        KeyError: If strict=True and unrecognized keys remain after conversion.
        ValueError: If file format is not recognized.
    """
    from pathlib import Path as _Path

    from safetensors.torch import load_file

    p = _Path(path)
    if p.suffix == ".safetensors":
        raw = load_file(str(p))
    elif p.suffix in (".pt", ".bin", ".pth"):
        raw = torch.load(str(p), map_location="cpu")
    else:
        raise ValueError(
            f"Unsupported checkpoint extension: {p.suffix}. "
            "Supported: .safetensors, .bin, .pt, .pth"
        )

    fmt = detect_format(raw)
    logger.info(f"Detected checkpoint format: '{fmt}' from {p.name}")

    converted = convert_state_dict(
        raw,
        source_format=fmt,
        num_heads=num_heads,
        hidden_size=hidden_size,
        num_double_blocks=num_double_blocks,
        num_single_blocks=num_single_blocks,
    )
    return converted
