"""Training logging utilities with WandB and TensorBoard support."""

import logging
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name (typically __name__).
        level: Logging level.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


class TrainingLogger:
    """Unified training logger supporting WandB and TensorBoard."""

    def __init__(self, config: DictConfig):
        """Initialize training logger.

        Args:
            config: Training configuration.
        """
        self.config = config
        self.logger = get_logger("training")

        self._wandb = None
        self._tensorboard = None

        # Setup logging backends
        self._setup_backends()

    def _setup_backends(self) -> None:
        """Setup logging backends based on config."""
        logging_config = self.config.get("logging", {})

        # WandB
        if logging_config.get("wandb", {}).get("enabled", False):
            self._setup_wandb(logging_config.wandb)

        # TensorBoard
        if logging_config.get("tensorboard", {}).get("enabled", False):
            self._setup_tensorboard(logging_config.tensorboard)

    def _setup_wandb(self, wandb_config: DictConfig) -> None:
        """Setup Weights & Biases logging."""
        try:
            import wandb

            wandb.init(
                project=wandb_config.get("project", "diff-base"),
                name=self.config.experiment.name,
                config=OmegaConf.to_container(self.config, resolve=True),
                resume=wandb_config.get("resume", "allow"),
                tags=wandb_config.get("tags", []),
            )
            self._wandb = wandb
            self.logger.info("WandB logging initialized")
        except ImportError:
            self.logger.warning("wandb not installed, skipping WandB logging")
        except Exception as e:
            self.logger.warning(f"Failed to initialize WandB: {e}")

    def _setup_tensorboard(self, tb_config: DictConfig) -> None:
        """Setup TensorBoard logging."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = Path(self.config.experiment.output_dir) / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)

            self._tensorboard = SummaryWriter(log_dir=str(log_dir))
            self.logger.info(f"TensorBoard logging initialized at {log_dir}")
        except ImportError:
            self.logger.warning("tensorboard not installed, skipping TensorBoard")
        except Exception as e:
            self.logger.warning(f"Failed to initialize TensorBoard: {e}")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to all backends.

        Args:
            metrics: Dictionary of metric names to values.
            step: Global step number.
        """
        # Console logging
        metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in metrics.items())
        self.logger.info(f"Step {step}: {metrics_str}")

        # WandB
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

        # TensorBoard
        if self._tensorboard is not None and step is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._tensorboard.add_scalar(key, value, step)

    def log_images(
        self,
        images: dict[str, Any],
        step: int | None = None
    ) -> None:
        """Log images to backends.

        Args:
            images: Dictionary of image names to image tensors.
            step: Global step number.
        """
        if self._wandb is not None:
            import wandb
            wandb_images = {
                k: wandb.Image(v) for k, v in images.items()
            }
            self._wandb.log(wandb_images, step=step)

        if self._tensorboard is not None and step is not None:
            for key, image in images.items():
                self._tensorboard.add_image(key, image, step)

    def log_histogram(
        self,
        name: str,
        values: Any,
        step: int | None = None
    ) -> None:
        """Log histogram to backends.

        Args:
            name: Histogram name.
            values: Values to histogram.
            step: Global step number.
        """
        if self._wandb is not None:
            import wandb
            self._wandb.log({name: wandb.Histogram(values)}, step=step)

        if self._tensorboard is not None and step is not None:
            self._tensorboard.add_histogram(name, values, step)

    def log_model_graph(self, model: Any, input_sample: Any) -> None:
        """Log model architecture graph.

        Args:
            model: PyTorch model.
            input_sample: Sample input for tracing.
        """
        if self._tensorboard is not None:
            try:
                self._tensorboard.add_graph(model, input_sample)
            except Exception as e:
                self.logger.warning(f"Failed to log model graph: {e}")

    def finish(self) -> None:
        """Finalize logging and close backends."""
        if self._wandb is not None:
            self._wandb.finish()

        if self._tensorboard is not None:
            self._tensorboard.close()

        self.logger.info("Training logging finished")

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finish()


class MetricsTracker:
    """Track and compute running metrics."""

    def __init__(self, window_size: int = 100):
        """Initialize metrics tracker.

        Args:
            window_size: Size of moving average window.
        """
        self.window_size = window_size
        self._metrics: dict[str, list[float]] = {}

    def update(self, metrics: dict[str, float]) -> None:
        """Update tracked metrics.

        Args:
            metrics: Dictionary of metric values.
        """
        for key, value in metrics.items():
            if key not in self._metrics:
                self._metrics[key] = []
            self._metrics[key].append(value)

            # Keep only window_size recent values
            if len(self._metrics[key]) > self.window_size:
                self._metrics[key] = self._metrics[key][-self.window_size:]

    def get_average(self, key: str) -> float | None:
        """Get moving average for a metric.

        Args:
            key: Metric name.

        Returns:
            Moving average or None if metric not tracked.
        """
        if key not in self._metrics or not self._metrics[key]:
            return None
        return sum(self._metrics[key]) / len(self._metrics[key])

    def get_all_averages(self) -> dict[str, float]:
        """Get all metric averages.

        Returns:
            Dictionary of metric averages.
        """
        return {
            key: self.get_average(key)
            for key in self._metrics
            if self.get_average(key) is not None
        }

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self._metrics.clear()
