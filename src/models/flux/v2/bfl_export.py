"""Export FLUX.2 internal (HF-aligned) checkpoints to BFL native format.

Inverse of the BFL->internal conversion in weight_mapping.py.
Symmetric counterpart of src/models/flux/v1/bfl_export.py for FLUX.2.

Key differences from v1 export:
- Shared modulation keys (double_stream_modulation_img/txt, single_stream_modulation)
- time_guidance_embed instead of time_text_embed
- pooled_text_embed instead of time_text_embed.text_embedder
- SwiGLU feed-forward (ff.linear_in / ff.linear_out)
- Single blocks have fused to_qkv_mlp_proj (no split/merge needed)
- All weights are bias=False
"""

from pathlib import Path

import torch

from ....utils.logging import get_logger
from .weight_mapping import StateDict, _convert_internal_to_bfl

logger = get_logger(__name__)


def convert_internal_to_bfl(
    state_dict: StateDict,
) -> StateDict:
    """Convert an internal (HF-aligned) FLUX.2 state dict to BFL native format.

    Inverse of ``convert_state_dict(state_dict, target='bfl')`` in weight_mapping.py.

    Unlike FLUX.1, no QKV splitting/merging is required because FLUX.2 stores the
    fused ``to_qkv_mlp_proj`` in both BFL and internal formats.

    Args:
        state_dict: State dict in internal (HF-aligned) key naming.

    Returns:
        State dict with BFL native key naming.
    """
    return _convert_internal_to_bfl(state_dict)


def to_bfl_checkpoint(
    model_or_state_dict: "torch.nn.Module | StateDict",
    output_path: str | Path,
) -> Path:
    """Export a FLUX.2 transformer to a BFL-native safetensors checkpoint.

    Symmetric counterpart of ``src.models.flux.v1.bfl_export.to_bfl_checkpoint``.

    The exported file uses BFL native key naming and can be reloaded via
    ``load_flux2_checkpoint(path, target='internal')`` in weight_mapping.py.

    Args:
        model_or_state_dict: Either a ``Flux2Transformer`` (its ``state_dict()``
            is used) or a pre-extracted state dict in internal (HF-aligned) naming.
        output_path: Destination ``.safetensors`` file path.

    Returns:
        Path to the saved file.

    Example::

        to_bfl_checkpoint(model.transformer, "/checkpoints/flux2-finetune.safetensors")
    """
    import torch.nn as nn
    from safetensors.torch import save_file

    if isinstance(model_or_state_dict, nn.Module):
        sd = {k: v.cpu() for k, v in model_or_state_dict.state_dict().items()}
    else:
        sd = {k: v.cpu() for k, v in model_or_state_dict.items()}

    bfl_sd = convert_internal_to_bfl(sd)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(bfl_sd, str(out_path))
    logger.info(f"FLUX.2 BFL checkpoint saved to {out_path} ({len(bfl_sd)} keys)")
    return out_path
