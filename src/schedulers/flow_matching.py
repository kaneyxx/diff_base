"""Flow matching scheduler for Flux models."""


import torch
from omegaconf import DictConfig


class FlowMatchingScheduler:
    """Flow matching scheduler for rectified flow models (Flux).

    Implements the rectified flow formulation where the model learns
    to predict the velocity field that transports noise to data.
    """

    def __init__(self, config: DictConfig):
        """Initialize flow matching scheduler.

        Args:
            config: Scheduler configuration.
        """
        self.config = config

        self.num_train_timesteps = config.get("num_train_timesteps", 1000)
        self.shift = config.get("shift", 3.0)  # Flux uses shift=3
        self.base_shift = config.get("base_shift", 0.5)
        self.max_shift = config.get("max_shift", 1.15)

        # Timesteps will be set during inference
        self.timesteps = None
        self.sigmas = None
        self.num_inference_steps = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device | str = "cpu",
    ) -> None:
        """Set inference timesteps.

        Args:
            num_inference_steps: Number of inference steps.
            device: Target device.
        """
        self.num_inference_steps = num_inference_steps

        # Create timesteps
        timesteps = torch.linspace(1, 0, num_inference_steps + 1, device=device)

        # Apply shift schedule
        sigmas = self._shift_schedule(timesteps)

        self.timesteps = timesteps[:-1]
        self.sigmas = sigmas

    def _shift_schedule(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Apply shift to timestep schedule.

        Args:
            timesteps: Linear timesteps in [0, 1].

        Returns:
            Shifted timesteps.
        """
        # Flux shift formula
        return self.shift * timesteps / (1 + (self.shift - 1) * timesteps)

    def scale_noise(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Scale and combine sample with noise for training.

        In flow matching: x_t = (1-t) * x_0 + t * noise

        Args:
            sample: Clean samples (x_0).
            timestep: Timesteps in [0, 1].
            noise: Noise samples.

        Returns:
            Noisy samples (x_t).
        """
        # Ensure timestep has correct shape
        while timestep.dim() < sample.dim():
            timestep = timestep.unsqueeze(-1)

        timestep = timestep.to(sample.device, dtype=sample.dtype)

        noisy_sample = (1 - timestep) * sample + timestep * noise
        return noisy_sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples (alias for scale_noise).

        Args:
            original_samples: Clean samples.
            noise: Noise to add.
            timesteps: Timesteps (should be in [0, 1] for flow matching).

        Returns:
            Noisy samples.
        """
        # Convert integer timesteps to [0, 1] if needed
        if timesteps.max() > 1:
            timesteps = timesteps.float() / self.num_train_timesteps

        return self.scale_noise(original_samples, timesteps, noise)

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity target for training.

        In flow matching: v = noise - sample

        Args:
            sample: Clean samples (x_0).
            noise: Noise.
            timesteps: Timesteps (unused for flow matching).

        Returns:
            Velocity targets.
        """
        return noise - sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform one denoising step.

        Args:
            model_output: Predicted velocity.
            timestep: Current timestep.
            sample: Current sample (x_t).
            generator: Random generator (unused).

        Returns:
            Tuple of (previous sample, predicted x_0).
        """
        # Get step index
        if self.timesteps is not None:
            step_idx = (self.timesteps == timestep).nonzero()
            if len(step_idx) > 0:
                step_idx = step_idx.item()
                dt = self.sigmas[step_idx] - self.sigmas[step_idx + 1]
            else:
                dt = 1.0 / self.num_inference_steps
        else:
            dt = 1.0 / self.num_train_timesteps

        # Euler step: x_{t-dt} = x_t - dt * v
        prev_sample = sample - dt * model_output

        # Estimate x_0 from velocity
        # x_t = (1-t) * x_0 + t * noise
        # v = noise - x_0
        # x_0 = x_t - t * v
        if isinstance(timestep, torch.Tensor):
            t = timestep.item() if timestep.dim() == 0 else timestep[0].item()
        else:
            t = timestep

        pred_original_sample = sample - t * model_output

        return prev_sample, pred_original_sample

    def get_sigmas(
        self,
        num_inference_steps: int,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Get sigma schedule for inference.

        Args:
            num_inference_steps: Number of steps.
            device: Target device.

        Returns:
            Sigma schedule.
        """
        timesteps = torch.linspace(1, 0, num_inference_steps + 1, device=device)
        return self._shift_schedule(timesteps)

    # ------------------------------------------------------------------
    # BFL-aligned methods (additive — existing methods above are unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def _mu(
        image_seq_len: int,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        x1: int = 256,
        x2: int = 4096,
    ) -> float:
        """Linear interpolation of mu by image sequence length.

        Matches BFL ``get_lin_function`` from ``src/flux/sampling.py``.

        Args:
            image_seq_len: Number of image tokens (H/patch * W/patch).
            base_shift: Shift value at ``x1`` sequence length.
            max_shift: Shift value at ``x2`` sequence length.
            x1: Lower anchor sequence length (default 256 = 16×16 at patch=2).
            x2: Upper anchor sequence length (default 4096 = 64×64 at patch=2).

        Returns:
            Scalar mu value for ``_time_shift``.
        """
        m = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - m * x1
        return image_seq_len * m + b

    @staticmethod
    def _time_shift(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
        """BFL time_shift: ``exp(mu) / (exp(mu) + (1/t - 1)**sigma)``.

        Matches BFL ``time_shift`` from ``src/flux/sampling.py`` at commit 4a3a3eb.
        At t=0, the limit is 0.0 and is set explicitly to avoid division by zero.

        Velocity target convention: ``v = noise - x_0`` (rectified flow).

        Args:
            mu: Resolution-dependent shift parameter (from ``_mu``).
            sigma: Exponent (always 1.0 in BFL training path).
            t: Timesteps tensor in [0, 1].

        Returns:
            Shifted timesteps tensor, same shape as ``t``.
        """
        exp_mu = torch.exp(torch.tensor(mu, dtype=t.dtype, device=t.device))
        # Avoid division by zero at t=0; limit is 0.0
        safe_t = t.clamp(min=torch.finfo(t.dtype).tiny)
        shifted = exp_mu / (exp_mu + (1.0 / safe_t - 1.0) ** sigma)
        return torch.where(t == 0.0, torch.zeros_like(t), shifted)

    def get_schedule(
        self,
        num_steps: int,
        image_seq_len: int,
        shift: bool = True,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Inference-time timestep schedule, BFL-aligned.

        Matches BFL ``get_schedule`` from ``src/flux/sampling.py`` at commit 4a3a3eb.

        Args:
            num_steps: Number of denoising steps.
            image_seq_len: Number of image tokens (determines shift mu).
            shift: If True, apply BFL resolution-aware time shift.
            base_shift: Base shift value (BFL default 0.5).
            max_shift: Max shift value (BFL default 1.15).
            device: Target device.

        Returns:
            Timestep tensor of shape ``[num_steps + 1]`` in descending order.
        """
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        if shift:
            mu = self._mu(image_seq_len, base_shift, max_shift)
            timesteps = self._time_shift(mu, 1.0, timesteps)
        return timesteps

    def training_sample(
        self,
        batch_size: int,
        image_seq_len: int,
        shift: bool = True,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ) -> torch.Tensor:
        """Sample per-sample timesteps for training.

        Draws ``t ~ Uniform(0, 1)`` then applies BFL time-shift if requested.

        Args:
            batch_size: Number of samples in the batch.
            image_seq_len: Number of image tokens (for shift mu computation).
            shift: If True, apply BFL resolution-aware time shift to sampled ``t``.
            device: Target device.
            generator: Optional RNG for reproducibility.
            dtype: Floating-point dtype for the output tensor.
            base_shift: Base shift value (BFL default 0.5).
            max_shift: Max shift value (BFL default 1.15).

        Returns:
            Timestep tensor of shape ``[batch_size]`` in ``(0, 1]``.
        """
        t = torch.rand(batch_size, device=device, generator=generator, dtype=dtype)
        if shift:
            mu = self._mu(image_seq_len, base_shift, max_shift)
            t = self._time_shift(mu, 1.0, t)
        return t

    def add_noise_to_target(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build noisy latent and velocity target for flow-matching training.

        Rectified flow formulation:
        - ``x_t = (1 - t) * x_0 + t * noise``
        - ``target_velocity = noise - x_0``

        Args:
            x_0: Clean latents of shape ``[B, ...]``.
            noise: Noise tensor, same shape as ``x_0``.
            t: Per-sample timesteps of shape ``[B]``.

        Returns:
            Tuple of ``(x_t, target_velocity)`` both of the same shape as ``x_0``.
        """
        t_b = t.view(-1, *([1] * (x_0.ndim - 1))).to(x_0.device, dtype=x_0.dtype)
        x_t = (1.0 - t_b) * x_0 + t_b * noise
        target_velocity = noise - x_0
        return x_t, target_velocity


class FluxFlowMatchingScheduler(FlowMatchingScheduler):
    """Specialized flow matching scheduler for Flux models.

    Includes guidance embedding computation and resolution-aware scheduling.
    """

    def __init__(self, config: DictConfig):
        """Initialize Flux scheduler."""
        super().__init__(config)

        self.guidance_scale = config.get("guidance_scale", 3.5)

    def get_sigmas_resolution_aware(
        self,
        num_inference_steps: int,
        height: int,
        width: int,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Get resolution-aware sigma schedule.

        Flux adjusts the shift based on resolution.

        Args:
            num_inference_steps: Number of steps.
            height: Image height.
            width: Image width.
            device: Target device.

        Returns:
            Sigma schedule.
        """
        # Calculate image sequence length
        image_seq_len = (height // 16) * (width // 16)

        # Adjust shift based on resolution
        # mu = base_shift + (max_shift - base_shift) * resolution_factor
        mu = self.base_shift + (self.max_shift - self.base_shift) * (
            image_seq_len / (1024 * 1024 / 256)
        )

        # Create shifted schedule
        timesteps = torch.linspace(1, 0, num_inference_steps + 1, device=device)

        # Apply resolution-aware shift
        sigmas = mu * timesteps / (1 + (mu - 1) * timesteps)

        return sigmas

    def get_guidance_embeds(
        self,
        guidance_scale: float,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create guidance scale embeddings.

        Args:
            guidance_scale: Classifier-free guidance scale.
            batch_size: Batch size.
            device: Target device.
            dtype: Data type.

        Returns:
            Guidance embeddings.
        """
        guidance = torch.full(
            (batch_size,),
            guidance_scale,
            device=device,
            dtype=dtype,
        )
        return guidance

    def prepare_timesteps(
        self,
        num_inference_steps: int,
        height: int,
        width: int,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare timesteps and sigmas for inference.

        Args:
            num_inference_steps: Number of steps.
            height: Target height.
            width: Target width.
            device: Target device.

        Returns:
            Tuple of (timesteps, sigmas).
        """
        sigmas = self.get_sigmas_resolution_aware(
            num_inference_steps, height, width, device
        )

        self.sigmas = sigmas
        self.timesteps = sigmas[:-1]
        self.num_inference_steps = num_inference_steps

        return self.timesteps, self.sigmas
