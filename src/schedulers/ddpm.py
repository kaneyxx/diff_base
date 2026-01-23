"""DDPM (Denoising Diffusion Probabilistic Models) scheduler."""

import math
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig


class DDPMScheduler:
    """DDPM noise scheduler for training and inference.

    Implements the noise schedule and methods for adding/removing noise
    as described in "Denoising Diffusion Probabilistic Models".
    """

    def __init__(self, config: DictConfig):
        """Initialize DDPM scheduler.

        Args:
            config: Scheduler configuration.
        """
        self.config = config

        self.num_train_timesteps = config.get("num_train_timesteps", 1000)
        self.beta_start = config.get("beta_start", 0.00085)
        self.beta_end = config.get("beta_end", 0.012)
        self.beta_schedule = config.get("beta_schedule", "scaled_linear")
        self.prediction_type = config.get("prediction_type", "epsilon")

        # Compute betas
        self.betas = self._get_betas()

        # Compute alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]),
            self.alphas_cumprod[:-1]
        ])

        # Precompute values for diffusion
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For v-prediction
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Inference timesteps
        self.timesteps = torch.arange(self.num_train_timesteps - 1, -1, -1)

    def _get_betas(self) -> torch.Tensor:
        """Compute beta schedule.

        Returns:
            Beta values for each timestep.
        """
        if self.beta_schedule == "linear":
            return torch.linspace(
                self.beta_start, self.beta_end, self.num_train_timesteps
            )
        elif self.beta_schedule == "scaled_linear":
            # Used in Stable Diffusion
            return torch.linspace(
                self.beta_start ** 0.5,
                self.beta_end ** 0.5,
                self.num_train_timesteps,
            ) ** 2
        elif self.beta_schedule == "squaredcos_cap_v2":
            # Cosine schedule
            return self._betas_for_alpha_bar(
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            )
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

    def _betas_for_alpha_bar(self, alpha_bar_fn, max_beta: float = 0.999):
        """Create beta schedule from alpha bar function."""
        betas = []
        for i in range(self.num_train_timesteps):
            t1 = i / self.num_train_timesteps
            t2 = (i + 1) / self.num_train_timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        return torch.tensor(betas)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples at given timesteps.

        Args:
            original_samples: Clean samples [B, C, H, W].
            noise: Noise to add [B, C, H, W].
            timesteps: Timesteps for each sample [B].

        Returns:
            Noisy samples.
        """
        # Move schedule to correct device and dtype
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        sqrt_alpha_prod = sqrt_alpha_prod.to(original_samples.device, dtype=original_samples.dtype)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.to(original_samples.device, dtype=original_samples.dtype)

        # Reshape for broadcasting
        while sqrt_alpha_prod.dim() < original_samples.dim():
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

        return noisy_samples

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity target for v-prediction.

        Args:
            sample: Clean samples.
            noise: Noise.
            timesteps: Timesteps.

        Returns:
            Velocity targets.
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        sqrt_alpha_prod = sqrt_alpha_prod.to(sample.device, dtype=sample.dtype)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.to(sample.device, dtype=sample.dtype)

        while sqrt_alpha_prod.dim() < sample.dim():
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one denoising step.

        Args:
            model_output: Model prediction.
            timestep: Current timestep.
            sample: Current noisy sample.
            generator: Random generator.

        Returns:
            Tuple of (previous sample, predicted original sample).
        """
        t = timestep

        # Get parameters for this timestep
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # Convert prediction type to epsilon
        if self.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t.sqrt() * model_output
            ) / alpha_prod_t.sqrt()
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (
                alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
            )
        else:
            pred_original_sample = model_output

        # Compute coefficients
        pred_original_sample_coeff = (
            alpha_prod_t_prev.sqrt() * self.betas[t]
        ) / beta_prod_t
        current_sample_coeff = (
            self.alphas[t].sqrt() * beta_prod_t_prev
        ) / beta_prod_t

        # Compute predicted previous sample
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * sample
        )

        # Add noise for timesteps > 0
        if t > 0:
            variance = self._get_variance(t)
            noise = torch.randn(
                sample.shape,
                generator=generator,
                device=sample.device,
                dtype=sample.dtype,
            )
            pred_prev_sample = pred_prev_sample + variance.sqrt() * noise

        return pred_prev_sample, pred_original_sample

    def _get_variance(self, t: int) -> torch.Tensor:
        """Get variance for timestep."""
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t]

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[t]
        return variance

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Set inference timesteps.

        Args:
            num_inference_steps: Number of inference steps.
        """
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = torch.arange(0, num_inference_steps) * step_ratio
        timesteps = timesteps.flip(0)
        self.timesteps = timesteps

    def get_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get signal-to-noise ratio for timesteps.

        Args:
            timesteps: Timestep values.

        Returns:
            SNR values.
        """
        alpha_cumprod = self.alphas_cumprod[timesteps]
        snr = alpha_cumprod / (1 - alpha_cumprod)
        return snr.to(timesteps.device)
