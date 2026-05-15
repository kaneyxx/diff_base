"""FLUX.2 Kontext + Fill end-to-end forward tests (US-3).

Tests run on CPU with a tiny Flux2Transformer stub:
- resolution 64 (latent 8x8 after VAE-equivalent /8 downscale)
- hidden_size 64, 1 double block + 1 single block
- latent_channels=4 for speed (in_channels = 4 * patch_size^2 = 16)

No GPU or HuggingFace downloads required.
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Tiny Flux2Transformer config helpers
# ---------------------------------------------------------------------------

# head_dim = hidden_size / num_attention_heads = 64 / 2 = 32
# sum(axes_dims_rope) must equal head_dim = 32
# FLUX.2 default is 4D axes summing to head_dim
_TINY_CFG = OmegaConf.create({
    "hidden_size": 64,
    "num_attention_heads": 2,
    "num_layers": 1,
    "num_single_layers": 1,
    # latent_channels=4, patch_size=2 → in_channels = 4 * 2 * 2 = 16
    "in_channels": 16,
    "joint_attention_dim": 64,
    "pooled_projection_dim": 64,
    "guidance_embeds": True,
    "axes_dims_rope": [8, 8, 8, 8],  # sum=32 == head_dim
    "rope_theta": 2000.0,
    "mlp_ratio": 2.0,
})

_LATENT_CHANNELS = 4  # C for the latent space
_PATCH_SIZE = 2


def _make_tiny_transformer(fill_extra_channels: int = 0):
    """Build a tiny Flux2Transformer on CPU."""
    from src.models.flux.v2.transformer import Flux2Transformer

    cfg = OmegaConf.merge(_TINY_CFG, OmegaConf.create({
        "fill_extra_channels": fill_extra_channels,
    }))
    model = Flux2Transformer(cfg, variant="dev")
    model.eval()
    return model


def _latent_seq(B: int, H_lat: int, W_lat: int) -> torch.Tensor:
    """Build a patchified latent sequence [B, seq, in_channels]."""
    from src.models.flux.v2.conditioning import rearrange_latent_to_sequence

    latent = torch.randn(B, _LATENT_CHANNELS, H_lat, W_lat)
    return rearrange_latent_to_sequence(latent, patch_size=_PATCH_SIZE)


def _fake_text(B: int, txt_seq: int = 4, txt_dim: int = 64, pool_dim: int = 64):
    return (
        torch.randn(B, txt_seq, txt_dim),  # encoder_hidden_states
        torch.randn(B, pool_dim),          # pooled_projections
    )


# ---------------------------------------------------------------------------
# Fake VAE that returns deterministic latents without any network call
# ---------------------------------------------------------------------------

class _FakeVAE(nn.Module):
    """Minimal VAE stub: encode returns a fixed-size latent, no scaling."""

    scaling_factor: float = 1.0
    shift_factor: float = 0.0

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.shape
        # Simulate /8 spatial downscale with latent_channels=4
        lat_h, lat_w = H // 8, W // 8
        return torch.randn(B, _LATENT_CHANNELS, lat_h, lat_w)


# ---------------------------------------------------------------------------
# Test 1: Kontext forward returns target-only sequence length
# ---------------------------------------------------------------------------

class TestKontextTargetOnlySlice:
    """AC: output seq length == target-only, reference tokens removed."""

    def test_kontext_forward_target_only_slice(self):
        from src.models.flux.v2.conditioning import prepare_kontext_conditioning

        B = 1
        # Target latent: 4x4 pixels, patch_size=2 → (4/2)*(4/2) = 4 patches
        H_lat, W_lat = 4, 4
        # After patchification: seq = (H_lat // patch_size) * (W_lat // patch_size)
        target_seq = (H_lat // _PATCH_SIZE) * (W_lat // _PATCH_SIZE)  # 4

        transformer = _make_tiny_transformer()

        # Build target hidden states (already patchified: [B, target_seq, in_channels])
        hidden_states = _latent_seq(B, H_lat, W_lat)
        assert hidden_states.shape == (B, target_seq, _LATENT_CHANNELS * _PATCH_SIZE * _PATCH_SIZE)
        enc_hs, pooled = _fake_text(B)

        # Build reference conditioning via prepare_kontext_conditioning
        # Use fake VAE: ref image 32x32 → latent 4x4 → same seq=4 patches
        ref_image = torch.randn(B, 3, 32, 32)  # 32/8=4 lat resolution
        fake_vae = _FakeVAE()
        img_cond_seq, img_cond_seq_ids = prepare_kontext_conditioning(
            reference_images=ref_image,
            vae=fake_vae,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        with torch.no_grad():
            out = transformer(
                hidden_states=hidden_states,
                timestep=torch.rand(B),
                encoder_hidden_states=enc_hs,
                pooled_projections=pooled,
                guidance=torch.ones(B) * 3.5,
                img_cond_seq=img_cond_seq,
                img_cond_seq_ids=img_cond_seq_ids,
            )

        # Output must equal target-only sequence length
        assert out.shape[1] == target_seq, (
            f"Expected output seq_len={target_seq} (target only), got {out.shape[1]}. "
            "Reference tokens must be sliced off before returning."
        )
        assert out.shape[1] != target_seq + img_cond_seq.shape[1], (
            "Output still includes reference tokens — slicing is broken."
        )


# ---------------------------------------------------------------------------
# Test 2: Kontext forward output is reshapeable to target latent shape
# ---------------------------------------------------------------------------

class TestKontextOutputShapeMatchesTargetLatent:
    """AC: Kontext output can be reshaped to [B, C, H/ph, W/ph]."""

    def test_kontext_forward_output_shape_matches_target_latent(self):
        from src.models.flux.v2.conditioning import (
            prepare_kontext_conditioning,
            rearrange_sequence_to_latent,
        )

        B = 1
        H_lat, W_lat = 4, 4
        # target_seq after patchification = (H_lat//patch)*(W_lat//patch)
        target_seq = (H_lat // _PATCH_SIZE) * (W_lat // _PATCH_SIZE)  # 4  # noqa: F841 — kept for inline documentation

        transformer = _make_tiny_transformer()
        hidden_states = _latent_seq(B, H_lat, W_lat)
        enc_hs, pooled = _fake_text(B)

        ref_image = torch.randn(B, 3, 32, 32)
        fake_vae = _FakeVAE()
        img_cond_seq, img_cond_seq_ids = prepare_kontext_conditioning(
            reference_images=ref_image,
            vae=fake_vae,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        with torch.no_grad():
            out = transformer(
                hidden_states=hidden_states,
                timestep=torch.rand(B),
                encoder_hidden_states=enc_hs,
                pooled_projections=pooled,
                guidance=torch.ones(B) * 3.5,
                img_cond_seq=img_cond_seq,
                img_cond_seq_ids=img_cond_seq_ids,
            )

        # Should reshape back to target latent [B, C, H_lat, W_lat] without error
        h_patches = H_lat // _PATCH_SIZE
        w_patches = W_lat // _PATCH_SIZE
        reshaped = rearrange_sequence_to_latent(
            out,
            height=h_patches,
            width=w_patches,
            channels=_LATENT_CHANNELS,
            patch_size=_PATCH_SIZE,
        )
        assert reshaped.shape == (B, _LATENT_CHANNELS, H_lat, W_lat), (
            f"Expected reshaped latent {(B, _LATENT_CHANNELS, H_lat, W_lat)}, "
            f"got {reshaped.shape}"
        )


# ---------------------------------------------------------------------------
# Test 3: Fill forward accepts extra channels without shape error
# ---------------------------------------------------------------------------

class TestFillForwardAcceptsExtraChannels:
    """AC: Flux2Transformer with fill_extra_channels>0 accepts img_cond input."""

    def test_fill_forward_accepts_extra_channels(self):
        from src.models.flux.v2.conditioning import prepare_fill_conditioning

        B = 1
        H_lat, W_lat = 4, 4
        in_channels = _LATENT_CHANNELS * _PATCH_SIZE * _PATCH_SIZE  # 16  # noqa: F841 — kept for inline documentation

        # Build fill conditioning using a fake VAE first to know img_cond shape
        ref_image = torch.randn(B, 3, 32, 32)
        # mask: 1 = inpaint region (top half)
        mask = torch.zeros(B, 1, 32, 32)
        mask[:, :, :16, :] = 1.0

        fake_vae = _FakeVAE()
        img_cond = prepare_fill_conditioning(
            reference_image=ref_image,
            mask=mask,
            vae=fake_vae,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # img_cond last dim = latent_dim + mask_dim = C*ph*pw + ph*pw
        # = 4*4 + 4 = 20
        # The transformer's x_embedder_fill takes (in_channels + fill_extra_channels)
        # where fill_extra_channels = img_cond.shape[-1] (total conditioning channels)
        fill_extra_channels = img_cond.shape[-1]  # 20
        assert fill_extra_channels > 0, "fill_extra_channels must be positive"

        transformer = _make_tiny_transformer(fill_extra_channels=fill_extra_channels)

        hidden_states = _latent_seq(B, H_lat, W_lat)
        enc_hs, pooled = _fake_text(B)

        # Should not raise any shape errors
        with torch.no_grad():
            out = transformer(
                hidden_states=hidden_states,
                timestep=torch.rand(B),
                encoder_hidden_states=enc_hs,
                pooled_projections=pooled,
                guidance=torch.ones(B) * 3.5,
                img_cond=img_cond,
            )

        seq = (H_lat // _PATCH_SIZE) * (W_lat // _PATCH_SIZE)
        # Fill mode does NOT change the output sequence length
        assert out.shape == (B, seq, transformer.in_channels), (
            f"Expected {(B, seq, transformer.in_channels)}, got {out.shape}"
        )


# ---------------------------------------------------------------------------
# Test 4: Fill mask correctly zeroes out inpaint region in conditioning latent
# ---------------------------------------------------------------------------

class TestFillMaskZeroedOutsideInpaintRegion:
    """AC: prepare_fill_conditioning zeroes masked_image in mask=1 region.

    Per the docstring: masked_image = image * (1 - mask)
    - where mask=1: masked_image pixels = 0  (inpaint region, cond latent ≈ 0)
    - where mask=0: masked_image pixels = image (original, cond latent preserved)
    """

    def test_fill_mask_zeroed_outside_inpaint_region(self):
        from src.models.flux.v2.conditioning import prepare_fill_conditioning

        B = 1
        H_img, W_img = 16, 16  # small image for determinism
        H_lat, W_lat = H_img // 8, W_img // 8  # 2x2 latent  # noqa: F841 — kept for inline documentation

        # Use a fake VAE that returns the masked image directly (as latent proxy)
        # so we can reason about the masking without a real encoder.
        class _PassthroughVAE(nn.Module):
            """Returns the masked_image mean-pooled to latent_channels channels."""
            scaling_factor: float = 1.0
            shift_factor: float = 0.0

            def encode(self, image: torch.Tensor) -> torch.Tensor:
                B, C, H, W = image.shape
                lat_h, lat_w = H // 8, W // 8
                # Downsample and project to latent_channels=4 (average RGB)
                img_ds = torch.nn.functional.interpolate(
                    image, size=(lat_h, lat_w), mode="bilinear", align_corners=False
                )
                # Repeat channels to reach latent_channels=4
                img_ds = img_ds[:, :1, :, :].expand(B, _LATENT_CHANNELS, lat_h, lat_w)
                return img_ds

        # Fully saturated image: all ones in the non-masked region
        ref_image = torch.ones(B, 3, H_img, W_img)

        # mask=1 means "inpaint here" → masked_image = image*(1-mask) = 0 there
        mask = torch.zeros(B, 1, H_img, W_img)
        mask[:, :, :H_img // 2, :] = 1.0  # top half = inpaint (masked)

        vae = _PassthroughVAE()
        img_cond = prepare_fill_conditioning(
            reference_image=ref_image,
            mask=mask,
            vae=vae,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # img_cond shape: [B, seq, latent_dim + mask_dim]
        # latent_dim = _LATENT_CHANNELS * ph * pw = 4*4 = 16
        # mask_dim = 1 * ph * pw = 4
        latent_dim = _LATENT_CHANNELS * _PATCH_SIZE * _PATCH_SIZE

        # The masked_image passed to VAE encode has:
        #   - top half pixels = 0 (mask=1 region → inpaint)
        #   - bottom half pixels = 1 (mask=0 region → keep)
        # After the passthrough VAE (which preserves the signal), the latent
        # for the masked (top) region should be ~zero, and non-zero for unmasked.
        # Extract the latent portion only (first latent_dim channels)
        latent_seq = img_cond[..., :latent_dim]  # [B, seq, latent_dim]  # noqa: F841 — kept for downstream slicing reference

        # seq = (lat_h // patch) * (lat_w // patch) = 1 * 1 = 1 (2x2 latent, patch=2)
        # All pixels in this tiny latent come from both regions, so we test
        # using a larger image where regions are clearly separated.

        # For a clearer test: use 32x32 image → 4x4 latent → 4 patches
        ref_image_32 = torch.ones(B, 3, 32, 32)
        mask_32 = torch.zeros(B, 1, 32, 32)
        mask_32[:, :, :16, :] = 1.0  # top half masked

        img_cond_32 = prepare_fill_conditioning(
            reference_image=ref_image_32,
            mask=mask_32,
            vae=vae,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # The masked_image = image * (1 - mask):
        # - Top half (mask=1): pixel = 1 * (1-1) = 0 → VAE input = 0 → latent ≈ 0
        # - Bottom half (mask=0): pixel = 1 * (1-0) = 1 → VAE input = 1 → latent ≠ 0
        # Our passthrough VAE preserves this exactly (no shift/scale).
        # The lat 4x4, patch_size=2 → 2x2=4 patches; top 2 patches = masked, bottom 2 = unmasked
        latent_seq_32 = img_cond_32[..., :_LATENT_CHANNELS * _PATCH_SIZE * _PATCH_SIZE]

        # Top patches (first half of seq): should be zero (masked region)
        # Bottom patches (second half of seq): should be non-zero (kept region)
        seq_len = latent_seq_32.shape[1]
        half = seq_len // 2
        masked_patches = latent_seq_32[0, :half]   # top half patches
        kept_patches = latent_seq_32[0, half:]     # bottom half patches

        assert masked_patches.abs().sum().item() == 0.0, (
            f"Masked region (mask=1) should have zero latent, "
            f"got sum={masked_patches.abs().sum().item():.4f}"
        )
        assert kept_patches.abs().sum().item() > 0.0, (
            f"Kept region (mask=0) should have non-zero latent, "
            f"got sum={kept_patches.abs().sum().item():.4f}"
        )
