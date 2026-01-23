"""Euler discrete scheduler for diffusion inference."""

import math
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig


class EulerDiscreteScheduler:
    """Euler discrete scheduler for fast sampling.

    Implements Euler method for solving the probability flow ODE.
    """

    def __init__(self, config: DictConfig):
        """Initialize Euler scheduler.

        Args:
            config: Scheduler configuration.
        """
        self.config = config

        self.num_train_timesteps = config.get("num_train_timesteps", 1000)
        self.beta_start = config.get("beta_start", 0.00085)
        self.beta_end = config.get("beta_end", 0.012)
        self.beta_schedule = config.get("beta_schedule", "scaled_linear")
        self.prediction_type = config.get("prediction_type", "epsilon")

        # Compute betas and sigmas
        self.betas = self._get_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Compute sigmas for Euler method
        sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        self.sigmas = torch.cat([sigmas, torch.zeros(1)])

        # Initialize with training timesteps
        self.timesteps = torch.arange(self.num_train_timesteps - 1, -1, -1)
        self.num_inference_steps = None

    def _get_betas(self) -> torch.Tensor:
        """Compute beta schedule."""
        if self.beta_schedule == "linear":
            return torch.linspace(
                self.beta_start, self.beta_end, self.num_train_timesteps
            )
        elif self.beta_schedule == "scaled_linear":
            return torch.linspace(
                self.beta_start ** 0.5,
                self.beta_end ** 0.5,
                self.num_train_timesteps,
            ) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Set inference timesteps.

        Args:
            num_inference_steps: Number of inference steps.
        """
        self.num_inference_steps = num_inference_steps

        # Compute timesteps
        timesteps = torch.linspace(
            self.num_train_timesteps - 1,
            0,
            num_inference_steps,
        )
        self.timesteps = timesteps.round().long()

        # Compute sigmas for these timesteps
        sigmas = self.sigmas[self.timesteps]
        self.sigmas_interp = torch.cat([sigmas, torch.zeros(1)])

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        """Scale model input based on current sigma.

        Args:
            sample: Input sample.
            timestep: Current timestep.

        Returns:
            Scaled sample.
        """
        step_idx = (self.timesteps == timestep).nonzero().item()
        sigma = self.sigmas_interp[step_idx]
        sample = sample / ((sigma ** 2 + 1) ** 0.5)
        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one Euler step.

        Args:
            model_output: Model prediction.
            timestep: Current timestep.
            sample: Current sample.
            generator: Random generator (unused).
            return_dict: Whether to return dict.

        Returns:
            Tuple of (prev_sample, pred_original_sample).
        """
        step_idx = (self.timesteps == timestep).nonzero().item()

        sigma = self.sigmas_interp[step_idx]
        sigma_next = self.sigmas_interp[step_idx + 1]

        # Convert model output to denoised sample
        if self.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = model_output * (-sigma / (sigma ** 2 + 1) ** 0.5) + \
                                   sample / (sigma ** 2 + 1)
        else:
            pred_original_sample = model_output

        # Euler step
        derivative = (sample - pred_original_sample) / sigma
        dt = sigma_next - sigma
        prev_sample = sample + derivative * dt

        return prev_sample, pred_original_sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples (for training).

        Args:
            original_samples: Clean samples.
            noise: Noise to add.
            timesteps: Timesteps.

        Returns:
            Noisy samples.
        """
        sigmas = self.sigmas[timesteps].to(original_samples.device, dtype=original_samples.dtype)

        while sigmas.dim() < original_samples.dim():
            sigmas = sigmas.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigmas
        return noisy_samples


class EulerAncestralDiscreteScheduler(EulerDiscreteScheduler):
    """Euler ancestral sampler with stochastic sampling."""

    def __init__(self, config: DictConfig):
        """Initialize scheduler."""
        super().__init__(config)
        self.eta = config.get("eta", 1.0)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one ancestral Euler step.

        Args:
            model_output: Model prediction.
            timestep: Current timestep.
            sample: Current sample.
            generator: Random generator.
            return_dict: Whether to return dict.

        Returns:
            Tuple of (prev_sample, pred_original_sample).
        """
        step_idx = (self.timesteps == timestep).nonzero().item()

        sigma = self.sigmas_interp[step_idx]
        sigma_next = self.sigmas_interp[step_idx + 1]

        # Convert to denoised
        if self.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = model_output * (-sigma / (sigma ** 2 + 1) ** 0.5) + \
                                   sample / (sigma ** 2 + 1)
        else:
            pred_original_sample = model_output

        # Ancestral step with noise
        sigma_up = min(
            sigma_next,
            self.eta * (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2) ** 0.5
        )
        sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5

        derivative = (sample - pred_original_sample) / sigma
        dt = sigma_down - sigma
        prev_sample = sample + derivative * dt

        if sigma_next > 0 and self.eta > 0:
            noise = torch.randn(
                sample.shape,
                generator=generator,
                device=sample.device,
                dtype=sample.dtype,
            )
            prev_sample = prev_sample + noise * sigma_up

        return prev_sample, pred_original_sample
