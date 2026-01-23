"""Flow matching scheduler for Flux models."""

import math
from typing import Optional, Tuple

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
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
