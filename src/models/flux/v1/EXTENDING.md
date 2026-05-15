# Extending FLUX.1 Transformer

This document describes the extension APIs available in `Flux1Transformer` for
building downstream conditioning models (ControlNet-on-Kontext, custom adapters,
etc.) without forking the transformer.

> **Experimental** — the APIs described here may change between versions.
> Pin to a specific commit if you need stability.

---

## 1. Block Hooks

`Flux1Transformer.forward()` accepts an optional `block_hooks` argument that
lets you inject a delta into hidden states after each block runs.

### Signature

```python
block_hooks: Optional[Dict[Literal["joint", "single"], List[Callable]]]
```

Each callable has the signature:

```python
def my_hook(
    block_idx: int,
    hidden_states: torch.Tensor,   # [B, seq, hidden_size] — full seq (target + ref if Kontext)
    txt_hidden: torch.Tensor | None,  # [B, txt_seq, hidden_size] for joint blocks; None for single
    temb: torch.Tensor,            # [B, hidden_size] — time/guidance embedding
) -> torch.Tensor:                 # delta to ADD to hidden_states — must be same shape
    ...
```

The returned tensor is **added** to `hidden_states` in-place (residual style).
Returning `torch.zeros_like(hidden_states)` is a no-op.

Passing `block_hooks=None` (the default) produces **identical** output to not
passing the argument at all — full backward compatibility is guaranteed.

### Example: zero-init ControlNet adapter

```python
import torch
import torch.nn as nn
from src.models.flux.v1.transformer import Flux1Transformer

class SimpleControlAdapter(nn.Module):
    def __init__(self, hidden_size: int, control_channels: int):
        super().__init__()
        # Zero-init projection so it starts as a no-op
        self.proj = nn.Linear(control_channels, hidden_size)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, control_feat: torch.Tensor) -> torch.Tensor:
        return self.proj(control_feat)


# Build transformer and register adapter so it's part of the state_dict
transformer = Flux1Transformer(config, variant="kontext")
adapter = SimpleControlAdapter(hidden_size=3072, control_channels=512)
transformer.register_conditioning_module("ctrl_adapter", adapter)

# Precompute control features (e.g. from a ControlNet encoder)
control_features = encode_control_image(canny_image)  # [B, seq, 512]

# Build a hook closure that captures the precomputed features
def ctrl_hook(block_idx, hidden_states, txt_hidden, temb):
    if block_idx in {4, 8, 12}:  # inject at chosen blocks only
        return adapter(control_features)
    return torch.zeros_like(hidden_states)

# Forward with hook
output = transformer(
    hidden_states=noisy_latent,
    timestep=timesteps,
    encoder_hidden_states=text_embeds,
    pooled_projections=pooled_embeds,
    guidance=guidance,
    img_cond_seq=reference_latent,
    img_cond_seq_ids=reference_ids,
    block_hooks={"joint": [ctrl_hook]},
)
```

### Notes

- Hooks receive the **full** `hidden_states` (target + reference tokens when
  Kontext is active). If your adapter should only affect target tokens, slice
  with `hidden_states[:, :target_seq_len]` and return a zero-padded delta.
- Multiple hooks per block type are supported; they are applied sequentially.
- Raising inside a hook propagates naturally — wrap with try/except if needed.
- The `block_idx` for single blocks is the index within `single_transformer_blocks`
  (0-based), independent of joint block indices.

---

## 2. `register_conditioning_module`

```python
transformer.register_conditioning_module(name: str, module: nn.Module) -> None
```

Stores `module` in `self.conditioning_modules` (`nn.ModuleDict`), which means:

- **`state_dict()` / `load_state_dict()`**: module parameters appear under
  `conditioning_modules.<name>.*`, so they save and load alongside the
  transformer weights automatically.
- **`.to(device)` / `.cuda()` / `.half()`**: the module moves with the
  transformer — no separate bookkeeping needed.
- **`parameters()` / `named_parameters()`**: the module's parameters are
  enumerated by the transformer's optimizer setup.

### Raises

| Condition | Exception |
|-----------|-----------|
| `module` is not an `nn.Module` | `TypeError` |
| `name` is already registered | `ValueError` |

### Saving and loading

```python
# Save — conditioning_modules are included automatically
torch.save(transformer.state_dict(), "checkpoint.pt")

# Load — conditioning_modules must be registered before loading
transformer.register_conditioning_module("ctrl_adapter", adapter)
transformer.load_state_dict(torch.load("checkpoint.pt"))
```

---

## 3. Adding a New Conditioning Input Channel (Fill-mode style for FLUX.1)

FLUX.1 does not have a built-in Fill mode (FLUX.2 does), but you can approximate
channel-wise conditioning by subclassing `Flux1Transformer` and overriding the
`x_embedder` projection to accept additional channels:

```python
class Flux1FillTransformer(Flux1Transformer):
    def __init__(self, config, variant="dev", extra_in_channels: int = 17):
        super().__init__(config, variant=variant)
        total_in = self.in_channels + extra_in_channels
        # Replace x_embedder to accept [latent || mask || masked_image]
        self.x_embedder = nn.Linear(total_in, self.hidden_size)

    def forward(self, hidden_states, *, img_cond=None, **kwargs):
        if img_cond is not None:
            hidden_states = torch.cat([hidden_states, img_cond], dim=-1)
        return super().forward(hidden_states, **kwargs)
```

This pattern matches how FLUX.2 Fill works internally.

---

## 4. Contract Summary

| API | Status | Guarantee |
|-----|--------|-----------|
| `block_hooks=None` (default) | Stable | Identical output to pre-hooks forward |
| `block_hooks={"joint": [...]}` | Experimental | Signature may change |
| `block_hooks={"single": [...]}` | Experimental | Signature may change |
| `register_conditioning_module` | Experimental | Interface may change |
| Hook delta shape must match `hidden_states` | Required | Enforced at runtime |
| Hook must return `torch.Tensor` | Required | `TypeError` raised otherwise |
