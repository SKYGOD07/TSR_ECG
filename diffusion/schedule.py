"""DDPM noise schedule (Stage 1).

Implements the standard DDPM forward-process coefficients:

    beta_t      linear in [beta_start, beta_end]
    alpha_t     = 1 - beta_t
    alpha_bar_t = prod_{s<=t} alpha_s

and the closed-form forward step

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon,   epsilon ~ N(0, I)

Indexing convention
-------------------
Everything in THIS module is ZERO-BASED: ``t`` runs over ``0 .. T-1``.
The human-facing 1..T notation used in the experiments is converted in
``diffusion.forward_process``. Do not mix the two conventions.

The coefficient tables are registered as buffers, so ``schedule.to(device)``
moves them with the module and they are never rebuilt per batch.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Gather per-sample coefficients and reshape them for broadcasting.

    a:       [T]          coefficient table (lives on some device)
    t:       [B]          zero-based timestep indices, long
    x_shape: shape of the tensor the result will multiply, e.g. [B, C, L]

    returns: [B, 1, 1, ...] with ``len(x_shape) - 1`` trailing singleton dims.
    """
    b = t.shape[0]
    # gather requires index and source on the SAME device -- move the index to
    # the table's device rather than the table to the CPU.
    out = a.gather(-1, t.to(a.device))
    out = out.to(t.device)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DiffusionSchedule(nn.Module):
    """Precomputed DDPM schedule with a closed-form ``add_noise``."""

    def __init__(self, T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        super().__init__()
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end

        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    @property
    def device(self) -> torch.device:
        return self.betas.device

    def sample_t(self, batch_size: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Uniformly sample one zero-based timestep per batch element -> [B] long."""
        return torch.randint(
            0, self.T, (batch_size,), device=self.device, generator=generator, dtype=torch.long
        )

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion in closed form.

        x_0:   [B, C, L] clean signal, channel-first (for ECG: C = 12 leads)
        t:     [B] zero-based timesteps in ``0 .. T-1``
        noise: optional pre-sampled epsilon with the same shape as ``x_0``

        returns ``(x_t, epsilon)``, both exactly ``x_0.shape``.
        """
        if t.dim() != 1 or t.shape[0] != x_0.shape[0]:
            raise ValueError(
                f"t must be a 1-D tensor of length B={x_0.shape[0]}, got shape {tuple(t.shape)}"
            )
        if t.dtype != torch.long:
            t = t.long()
        t_min, t_max = int(t.min()), int(t.max())
        if t_min < 0 or t_max >= self.T:
            raise ValueError(
                f"zero-based timesteps must lie in [0, {self.T - 1}], got [{t_min}, {t_max}]"
            )

        if noise is None:
            noise = torch.randn_like(x_0)
        elif noise.shape != x_0.shape:
            raise ValueError(f"noise shape {tuple(noise.shape)} != x_0 shape {tuple(x_0.shape)}")

        sqrt_ab_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_ab_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        x_t = sqrt_ab_t * x_0 + sqrt_one_minus_ab_t * noise

        assert x_t.shape == x_0.shape, "forward diffusion must preserve shape"
        return x_t, noise

    def extra_repr(self) -> str:
        return f"T={self.T}, beta_start={self.beta_start}, beta_end={self.beta_end}"
