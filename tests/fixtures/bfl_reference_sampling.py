"""BFL reference sampling implementation.

Vendored from black-forest-labs/flux commit 4a3a3eb (src/flux/sampling.py).
Used as ground-truth in flow-matching alignment tests.

Provenance:
  Repository : https://github.com/black-forest-labs/flux
  Commit     : 4a3a3eb
  File       : src/flux/sampling.py
  Functions  : get_lin_function, time_shift, get_schedule
  License    : Apache 2.0 (as distributed in BFL repository)
"""

import math


def get_lin_function(
    x1: float = 256.0,
    y1: float = 0.5,
    x2: float = 4096.0,
    y2: float = 1.15,
):
    """Return a linear function f such that f(x1)=y1 and f(x2)=y2.

    BFL uses this to compute mu = f(image_seq_len).
    """
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    def linear_function(x: float) -> float:
        return m * x + b

    return linear_function


def time_shift(mu: float, sigma: float, t: float) -> float:
    """BFL time_shift scalar implementation.

    Formula: exp(mu) / (exp(mu) + (1/t - 1)**sigma).
    At t=0, the limit is 0.0 (numerator finite, denominator → ∞).

    Args:
        mu   : Resolution-dependent shift scalar.
        sigma: Exponent (always 1.0 in BFL training path).
        t    : Scalar timestep in [0, 1].

    Returns:
        Shifted timestep scalar.
    """
    if t == 0.0:
        return 0.0
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    """BFL inference-time timestep schedule.

    Args:
        num_steps     : Number of denoising steps.
        image_seq_len : Number of image tokens.
        base_shift    : Shift lower bound.
        max_shift     : Shift upper bound.
        shift         : If False, return uniform linspace.

    Returns:
        List of ``num_steps + 1`` timesteps descending from 1.0 to 0.0.
    """
    # linspace from 1 to 0 inclusive
    timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]
    if shift:
        mu_fn = get_lin_function(y1=base_shift, y2=max_shift)
        mu = mu_fn(image_seq_len)
        timesteps = [time_shift(mu, 1.0, t) for t in timesteps]
    return timesteps
