"""Export FLUX.1 internal (HF-aligned) checkpoints to BFL native format.

Inverse of the BFL->internal conversion in weight_mapping.py.
Fused QKV tensors in BFL format are reconstructed by concatenating split q/k/v
back along dim=0.
"""

from pathlib import Path
from typing import Literal

import torch

from ....utils.logging import get_logger
from .weight_mapping import StateDict

logger = get_logger(__name__)

ExportFormat = Literal["bfl", "diffusers", "both"]

# Inverse of STATIC_MAP in weight_mapping._convert_bfl_to_internal
_INTERNAL_TO_BFL_STATIC: dict[str, str] = {
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

# Inverse of DOUBLE_SUFFIX_MAP in weight_mapping._map_double_block_key
_INTERNAL_TO_BFL_DOUBLE: dict[str, str] = {
    "attn.to_out.0.weight": "img_attn.proj.weight",
    "attn.to_out.0.bias": "img_attn.proj.bias",
    "attn.to_add_out.weight": "txt_attn.proj.weight",
    "attn.to_add_out.bias": "txt_attn.proj.bias",
    "attn.norm_q.weight": "img_attn.norm.query_norm.scale",
    "attn.norm_k.weight": "img_attn.norm.key_norm.scale",
    "attn.norm_added_q.weight": "txt_attn.norm.query_norm.scale",
    "attn.norm_added_k.weight": "txt_attn.norm.key_norm.scale",
    "ff.net.0.proj.weight": "img_mlp.0.weight",
    "ff.net.0.proj.bias": "img_mlp.0.bias",
    "ff.net.2.weight": "img_mlp.2.weight",
    "ff.net.2.bias": "img_mlp.2.bias",
    "ff_context.net.0.proj.weight": "txt_mlp.0.weight",
    "ff_context.net.0.proj.bias": "txt_mlp.0.bias",
    "ff_context.net.2.weight": "txt_mlp.2.weight",
    "ff_context.net.2.bias": "txt_mlp.2.bias",
    "norm1.linear.weight": "img_mod.lin.weight",
    "norm1.linear.bias": "img_mod.lin.bias",
    "norm1_context.linear.weight": "txt_mod.lin.weight",
    "norm1_context.linear.bias": "txt_mod.lin.bias",
}

# Inverse of SINGLE_SUFFIX_MAP in weight_mapping._map_single_block_key
_INTERNAL_TO_BFL_SINGLE: dict[str, str] = {
    "proj_out.weight": "linear2.weight",
    "proj_out.bias": "linear2.bias",
    "norm.linear.weight": "modulation.lin.weight",
    "norm.linear.bias": "modulation.lin.bias",
    "attn.norm_q.weight": "pre_norm.query_norm.scale",
    "attn.norm_k.weight": "pre_norm.key_norm.scale",
}


def convert_internal_to_bfl(
    state_dict: StateDict,
    num_double_blocks: int = 19,
    num_single_blocks: int = 38,
) -> StateDict:
    """Convert an internal (HF-aligned) FLUX.1 state dict to BFL native format.

    Inverse of ``convert_state_dict(..., source_format="bfl")`` in weight_mapping.py.
    Split q/k/v tensors are concatenated back into fused qkv; split q/k/v/mlp
    in single blocks are fused back into linear1.

    Args:
        state_dict: State dict in internal (HF-aligned) key naming.
        num_double_blocks: Number of double (joint) transformer blocks.
        num_single_blocks: Number of single transformer blocks.

    Returns:
        State dict with BFL native key naming.

    Raises:
        KeyError: If required split components (to_q/to_k/to_v) are missing
            for a block that has fused qkv in BFL format.
    """
    out: StateDict = {}

    # --- Accumulators for fused tensors ---
    # double block: {block_idx: {weight|bias: {q, k, v}}}
    img_qkv: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    txt_qkv: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    # single block: {block_idx: {weight|bias: {q, k, v, mlp}}}
    single_qkv: dict[str, dict[str, dict[str, torch.Tensor]]] = {}

    import re

    for key, val in state_dict.items():
        # Static top-level keys
        if key in _INTERNAL_TO_BFL_STATIC:
            out[_INTERNAL_TO_BFL_STATIC[key]] = val
            continue

        # Double blocks: transformer_blocks.N.<suffix>
        m = re.match(r"^transformer_blocks\.(\d+)\.", key)
        if m:
            n = m.group(1)
            suffix = key[m.end():]
            bfl_prefix = f"double_blocks.{n}"

            # Fused img qkv components
            if suffix in ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"):
                part = suffix.split(".")[1]  # to_q, to_k, to_v
                img_qkv.setdefault(n, {}).setdefault("weight", {})[part] = val
                continue
            if suffix in ("attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias"):
                part = suffix.split(".")[1]  # to_q, to_k, to_v
                img_qkv.setdefault(n, {}).setdefault("bias", {})[part] = val
                continue

            # Fused txt qkv components
            if suffix in ("attn.add_q_proj.weight", "attn.add_k_proj.weight", "attn.add_v_proj.weight"):
                part = suffix.split(".")[1]  # add_q_proj, add_k_proj, add_v_proj
                txt_qkv.setdefault(n, {}).setdefault("weight", {})[part] = val
                continue
            if suffix in ("attn.add_q_proj.bias", "attn.add_k_proj.bias", "attn.add_v_proj.bias"):
                part = suffix.split(".")[1]
                txt_qkv.setdefault(n, {}).setdefault("bias", {})[part] = val
                continue

            # Direct renames
            if suffix in _INTERNAL_TO_BFL_DOUBLE:
                out[f"{bfl_prefix}.{_INTERNAL_TO_BFL_DOUBLE[suffix]}"] = val
                continue

            logger.warning(f"Unrecognized double block suffix: {suffix!r} (key={key!r})")
            continue

        # Single blocks: single_transformer_blocks.N.<suffix>
        m = re.match(r"^single_transformer_blocks\.(\d+)\.", key)
        if m:
            n = m.group(1)
            suffix = key[m.end():]
            bfl_prefix = f"single_blocks.{n}"

            # Fused linear1 components: q/k/v
            if suffix in ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"):
                part = suffix.split(".")[1]
                single_qkv.setdefault(n, {}).setdefault("weight", {})[part] = val
                continue
            if suffix in ("attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias"):
                part = suffix.split(".")[1]
                single_qkv.setdefault(n, {}).setdefault("bias", {})[part] = val
                continue
            # mlp part of linear1
            if suffix == "proj_mlp.weight":
                single_qkv.setdefault(n, {}).setdefault("weight", {})["mlp"] = val
                continue
            if suffix == "proj_mlp.bias":
                single_qkv.setdefault(n, {}).setdefault("bias", {})["mlp"] = val
                continue

            # Direct renames
            if suffix in _INTERNAL_TO_BFL_SINGLE:
                out[f"{bfl_prefix}.{_INTERNAL_TO_BFL_SINGLE[suffix]}"] = val
                continue

            logger.warning(f"Unrecognized single block suffix: {suffix!r} (key={key!r})")
            continue

        logger.warning(f"Unrecognized internal key: {key!r}")

    # --- Assemble fused double-block img qkv ---
    for n, dtype_map in img_qkv.items():
        bfl_prefix = f"double_blocks.{n}"
        for dtype_str, parts in dtype_map.items():
            fused = torch.cat([parts["to_q"], parts["to_k"], parts["to_v"]], dim=0)
            out[f"{bfl_prefix}.img_attn.qkv.{dtype_str}"] = fused

    # --- Assemble fused double-block txt qkv ---
    for n, dtype_map in txt_qkv.items():
        bfl_prefix = f"double_blocks.{n}"
        for dtype_str, parts in dtype_map.items():
            fused = torch.cat([parts["add_q_proj"], parts["add_k_proj"], parts["add_v_proj"]], dim=0)
            out[f"{bfl_prefix}.txt_attn.qkv.{dtype_str}"] = fused

    # --- Assemble fused single-block linear1 [q, k, v, mlp] ---
    for n, dtype_map in single_qkv.items():
        bfl_prefix = f"single_blocks.{n}"
        for dtype_str, parts in dtype_map.items():
            components = [parts["to_q"], parts["to_k"], parts["to_v"]]
            if "mlp" in parts:
                components.append(parts["mlp"])
            fused = torch.cat(components, dim=0)
            out[f"{bfl_prefix}.linear1.{dtype_str}"] = fused

    logger.info(f"Converted {len(state_dict)} internal keys to {len(out)} BFL keys")
    return out


def to_bfl_checkpoint(
    model_or_state_dict: "torch.nn.Module | StateDict",
    output_path: str | Path,
    num_double_blocks: int = 19,
    num_single_blocks: int = 38,
) -> Path:
    """Export a FLUX.1 transformer to a BFL-native safetensors checkpoint.

    The exported file is compatible with the BFL inference pipeline
    (``black-forest-labs/flux``) and can be loaded via ``Flux1Model.from_bfl_checkpoint()``.

    Args:
        model_or_state_dict: Either an ``nn.Module`` (its ``state_dict()`` is used)
            or a pre-extracted state dict in internal (HF-aligned) naming.
        output_path: Destination ``.safetensors`` file path.
        num_double_blocks: Number of double transformer blocks (default 19 for FLUX.1-dev).
        num_single_blocks: Number of single transformer blocks (default 38 for FLUX.1-dev).

    Returns:
        Path to the saved file.

    Example::

        to_bfl_checkpoint(model.transformer, "/checkpoints/flux1-finetune.safetensors")
    """
    import torch.nn as nn
    from safetensors.torch import save_file

    if isinstance(model_or_state_dict, nn.Module):
        sd = {k: v.cpu() for k, v in model_or_state_dict.state_dict().items()}
    else:
        sd = {k: v.cpu() for k, v in model_or_state_dict.items()}

    bfl_sd = convert_internal_to_bfl(
        sd,
        num_double_blocks=num_double_blocks,
        num_single_blocks=num_single_blocks,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(bfl_sd, str(out_path))
    logger.info(f"BFL checkpoint saved to {out_path} ({len(bfl_sd)} keys)")
    return out_path
