"""Kontext training infrastructure for FLUX.1 image-editing.

Provides:
- ``KontextTrainerMixin``: shared training-step logic for both LoRA and full fine-tune.
- ``KontextLoRATrainer``: LoRA trainer with Kontext step.
- ``KontextFullFinetuneTrainer``: Full fine-tune trainer with Kontext step.
"""


import tempfile

import torch
import torch.nn.functional as F  # noqa: N812
from omegaconf import DictConfig

from ..data.cache import EmbeddingCache
from ..models.flux.v1.conditioning import (
    create_position_ids,
    prepare_kontext_conditioning,
    rearrange_latent_to_sequence,
)
from ..utils.logging import get_logger
from .flux_full_finetune_trainer import FluxFullFinetuneTrainer
from .lora_trainer import LoRATrainer

logger = get_logger(__name__)


class KontextTrainerMixin:
    """Mixin that provides a Kontext-aware training step.

    Must be mixed into a class that also inherits from ``BaseTrainer`` (or a
    subclass). Concrete subclasses must call ``super().__init__(config)`` so
    that ``self.model``, ``self.noise_scheduler``, ``self.device``, and
    ``self.dtype`` are all populated before ``training_step`` is invoked.

    Loss masking rationale
    ----------------------
    ``Flux1Transformer.forward()`` returns only the **target-token** portion of
    the hidden states (Phase A slicing fix).  The reference tokens never appear
    in the model output, so computing MSE loss directly on the output tensor
    already enforces target-only loss — no additional masking is required at
    the loss level.
    """

    # ------------------------------------------------------------------
    # Text embedding cache helpers
    # ------------------------------------------------------------------

    def _init_text_cache(self, config: DictConfig) -> None:
        """Initialise ``self._text_cache`` when ``training.cache_text_embeddings`` is true.

        Call this from a concrete subclass ``__init__`` *after* the base
        ``__init__`` has run (so ``config`` is available).

        Args:
            config: Full training configuration (OmegaConf DictConfig).
        """
        if config.training.get("cache_text_embeddings", False):
            cache_dir = (
                config.data.get("cache_dir", None)
                if hasattr(config, "data")
                else None
            )
            if cache_dir is None:
                cache_dir = tempfile.mkdtemp(prefix="text_emb_cache_")
                logger.warning(
                    "data.cache_dir not set; using temporary directory %s for "
                    "text embedding cache (will be lost after process exits).",
                    cache_dir,
                )
            else:
                import os
                cache_dir = os.path.join(cache_dir, "text_embeddings")
            self._text_cache: EmbeddingCache | None = EmbeddingCache(cache_dir)
            logger.info("Text embedding cache enabled at %s", cache_dir)
        else:
            self._text_cache = None

    def _get_cached_text_output(
        self,
        captions: list[str],
        device: torch.device,
        model: object,
    ) -> dict[str, torch.Tensor]:
        """Return text embeddings, using ``self._text_cache`` to avoid re-encoding.

        For each caption in *captions*:
        - If a cached entry exists, use it directly (memory or disk hit).
        - Otherwise batch-encode the misses, store results in the cache.

        The returned tensors are moved to *device*.

        Args:
            captions: List of B caption strings.
            device: Target device for output tensors.
            model: Model with an ``encode_text(captions, device)`` method.

        Returns:
            Dict with keys ``"prompt_embeds"`` and optionally
            ``"pooled_prompt_embeds"``, each a batched tensor on *device*.
        """
        cache = self._text_cache  # guaranteed non-None when this is called

        hits: dict[int, dict[str, torch.Tensor]] = {}
        miss_indices: list[int] = []
        miss_captions: list[str] = []

        for i, caption in enumerate(captions):
            cached = cache.get(caption)  # type: ignore[union-attr]
            if cached is not None:
                hits[i] = cached
            else:
                miss_indices.append(i)
                miss_captions.append(caption)

        # Encode cache misses in a single batch call
        if miss_captions:
            with torch.no_grad():
                new_output = model.encode_text(miss_captions, device=device)  # type: ignore[attr-defined]
            for local_idx, global_idx in enumerate(miss_indices):
                entry = {
                    k: v[local_idx : local_idx + 1].cpu().contiguous()
                    for k, v in new_output.items()
                    if v is not None
                }
                cache.put(miss_captions[local_idx], entry)  # type: ignore[union-attr]
                hits[global_idx] = entry

        # Re-assemble in original order and move to device
        keys = list(hits[0].keys())
        result: dict[str, torch.Tensor] = {}
        for key in keys:
            result[key] = torch.cat(
                [hits[i][key].to(device) for i in range(len(captions))], dim=0
            )
        return result

    # ------------------------------------------------------------------
    # Core Kontext training step
    # ------------------------------------------------------------------

    def _kontext_training_step(self, batch: dict) -> torch.Tensor:
        """Perform one forward + loss computation for a Kontext batch.

        Steps:
        1. VAE-encode both target and reference pixel tensors.
        2. Sample a continuous timestep in ``[0, 1]``.
        3. Build the noisy target latent via flow matching.
        4. Build ``img_cond_seq`` / ``img_cond_seq_ids`` from the reference.
        5. Encode text.
        6. Forward through ``model.transformer`` (output is target-only).
        7. Return flow-matching velocity loss.

        Args:
            batch: Batched dict from ``kontext_collate_fn``. Expected keys:
                - ``target_pixel``: ``[B, 3, Ht, Wt]`` in ``[-1, 1]``.
                - ``reference_pixel``: ``[B, 3, Hr, Wr]`` in ``[-1, 1]``.
                - ``captions``: list of B strings.

        Returns:
            Scalar loss tensor.
        """
        target_pixel: torch.Tensor = batch["target_pixel"]
        reference_pixel: torch.Tensor = batch["reference_pixel"]
        captions: list[str] = batch["captions"]

        batch_size = target_pixel.shape[0]
        device = self.device  # type: ignore[attr-defined]
        dtype = self.dtype    # type: ignore[attr-defined]
        model = self.model    # type: ignore[attr-defined]

        # ------------------------------------------------------------------
        # 1. VAE-encode target → clean latent [B, C, H/8, W/8]
        # ------------------------------------------------------------------
        with torch.no_grad():
            target_latent = model.encode_image(target_pixel.to(device, dtype=dtype))

        # ------------------------------------------------------------------
        # 2. Sample continuous timestep in [0, 1]
        # ------------------------------------------------------------------
        flow_shift = self.config.training.get("flow_shift", False)  # type: ignore[attr-defined]

        if flow_shift and hasattr(self.noise_scheduler, "training_sample"):  # type: ignore[attr-defined]
            seq_len = model.compute_image_seq_len(target_latent)
            timesteps = self.noise_scheduler.training_sample(  # type: ignore[attr-defined]
                batch_size=batch_size,
                image_seq_len=seq_len,
                shift=True,
                device=device,
                dtype=dtype,
            )
        else:
            timesteps = torch.rand(batch_size, device=device, dtype=dtype)

        # ------------------------------------------------------------------
        # 3. Build noisy target via flow matching: x_t = (1-t)*x0 + t*noise
        # ------------------------------------------------------------------
        noise = torch.randn_like(target_latent)

        if flow_shift and hasattr(self.noise_scheduler, "add_noise_to_target"):  # type: ignore[attr-defined]
            noisy_target, velocity_target = self.noise_scheduler.add_noise_to_target(  # type: ignore[attr-defined]
                target_latent, noise, timesteps
            )
        else:
            noisy_target = self.noise_scheduler.scale_noise(  # type: ignore[attr-defined]
                target_latent, timesteps, noise
            )
            # Velocity target: v = noise - x0
            velocity_target = self.noise_scheduler.get_velocity(  # type: ignore[attr-defined]
                target_latent, noise, timesteps
            )

        # ------------------------------------------------------------------
        # 4. Kontext conditioning from reference image
        # ------------------------------------------------------------------
        with torch.no_grad():
            img_cond_seq, img_cond_seq_ids = prepare_kontext_conditioning(
                reference_images=reference_pixel.to(device, dtype=dtype),
                vae=model.vae,
                device=device,
                dtype=dtype,
            )

        # ------------------------------------------------------------------
        # 5. Text encoding (with optional embedding cache)
        # ------------------------------------------------------------------
        if getattr(self, "_text_cache", None) is not None:
            text_output = self._get_cached_text_output(captions, device, model)
        else:
            with torch.no_grad():
                text_output = model.encode_text(captions, device=device)
        encoder_hidden_states = text_output["prompt_embeds"]
        pooled_projections = text_output.get("pooled_prompt_embeds")

        # ------------------------------------------------------------------
        # 6. Flatten target latent [B, C, H, W] → [B, seq, C*patch²]
        # ------------------------------------------------------------------
        noisy_seq = rearrange_latent_to_sequence(noisy_target, patch_size=2)

        # ------------------------------------------------------------------
        # 7. Build target position IDs for the noisy target
        # ------------------------------------------------------------------
        _, _, h_lat, w_lat = noisy_target.shape
        h_pat, w_pat = h_lat // 2, w_lat // 2
        target_ids = create_position_ids(
            batch_size=batch_size,
            height=h_pat,
            width=w_pat,
            device=device,
            dtype=dtype,
            time_offset=0.0,  # target stream = 0
        )

        # ------------------------------------------------------------------
        # 8. Guidance tensor (dev variant only)
        # ------------------------------------------------------------------
        guidance: torch.Tensor | None = None
        if getattr(model, "use_guidance", False):
            guidance_scale = self.config.training.get(  # type: ignore[attr-defined]
                "guidance_scale", 3.5
            )
            guidance = torch.full(
                (batch_size,),
                guidance_scale,
                device=device,
                dtype=dtype,
            )

        # ------------------------------------------------------------------
        # 9. Transformer forward — output is target-token only (Phase A fix)
        # ------------------------------------------------------------------
        predicted_velocity = model.transformer(
            hidden_states=noisy_seq,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            guidance=guidance,
            img_ids=target_ids,
            img_cond_seq=img_cond_seq,
            img_cond_seq_ids=img_cond_seq_ids,
        )

        # ------------------------------------------------------------------
        # 10. Flow-matching MSE loss over target tokens only
        #     velocity_target shape: [B, C, H/8, W/8] → rearrange to sequence
        # ------------------------------------------------------------------
        velocity_target_seq = rearrange_latent_to_sequence(
            velocity_target, patch_size=2
        )
        loss = F.mse_loss(predicted_velocity, velocity_target_seq)
        return loss

    # ------------------------------------------------------------------
    # Public training_step override
    # ------------------------------------------------------------------

    def training_step(self, batch: dict) -> torch.Tensor:
        """Dispatch to the Kontext training step.

        Args:
            batch: Batched Kontext sample dict.

        Returns:
            Scalar loss tensor.
        """
        return self._kontext_training_step(batch)

    # ------------------------------------------------------------------
    # Dataloader override — Kontext requires paired (target, ref) samples
    # ------------------------------------------------------------------

    def _setup_dataloader(self):
        """Build a Kontext-compatible dataloader (paired target/reference).

        Routes through :func:`src.data.create_kontext_dataloader` which
        dispatches via ``config.data.dataset_type`` (default ``kontext``;
        register additional paired datasets like ``orion``, ``bci``).
        """
        from ..data import create_kontext_dataloader
        return create_kontext_dataloader(self.config)


# ---------------------------------------------------------------------------
# Concrete Kontext trainers
# ---------------------------------------------------------------------------


class KontextLoRATrainer(KontextTrainerMixin, LoRATrainer):
    """LoRA trainer specialised for FLUX.1 Kontext (paired image-editing).

    Inherits LoRA injection / frozen-base-model setup from ``LoRATrainer``
    and replaces the training step with the Kontext-aware mixin.

    Example::

        trainer = KontextLoRATrainer(config)
        trainer.train()
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize KontextLoRATrainer.

        Args:
            config: Full training configuration.
        """
        # LoRATrainer.__init__ sets up model, optimizer, scheduler, dataloader.
        LoRATrainer.__init__(self, config)
        self._init_text_cache(config)

    def training_step(self, batch: dict) -> torch.Tensor:
        """Kontext LoRA training step.

        Args:
            batch: Batched Kontext sample dict.

        Returns:
            Scalar loss tensor.
        """
        return self._kontext_training_step(batch)


class KontextFullFinetuneTrainer(KontextTrainerMixin, FluxFullFinetuneTrainer):
    """Full fine-tune trainer specialised for FLUX.1 Kontext.

    MRO: KontextTrainerMixin → FluxFullFinetuneTrainer → FullFinetuneTrainer → BaseTrainer.
    - ``training_step`` comes from ``KontextTrainerMixin`` (Kontext-aware step).
    - ``_setup_optimizer``, EMA, 8-bit Adam, schnell guard come from ``FluxFullFinetuneTrainer``.
    - ``KontextLoRATrainer`` is unaffected by this change.

    All transformer parameters are trainable; VAE and text encoders are frozen.

    Example::

        trainer = KontextFullFinetuneTrainer(config)
        trainer.train()
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize KontextFullFinetuneTrainer.

        Args:
            config: Full training configuration.
        """
        FluxFullFinetuneTrainer.__init__(self, config)
        self._init_text_cache(config)

    def training_step(self, batch: dict) -> torch.Tensor:
        """Kontext full fine-tune training step.

        Args:
            batch: Batched Kontext sample dict.

        Returns:
            Scalar loss tensor.
        """
        return self._kontext_training_step(batch)
