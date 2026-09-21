"""Diffusion timestep embedding (Stage 2).

Deliberately the simple MLP from the design note -- ``Linear(1, dim) -> SiLU ->
Linear(dim, dim)`` -- not a sinusoidal embedding.  Its only job is to tell the
reused TSR encoder WHICH diffusion step produced ``x_t``.

The one addition over the sketch is ``max_t``: a raw timestep index of up to
100 entering a ``Linear(1, dim)`` gives the first layer a very large input
scale relative to its initialisation, so ``t`` is divided by ``max_t`` to land
in ``(0, 1]``.  That is input scaling, not a change of embedding family.
"""

from typing import Optional

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """Map a batch of timesteps ``[B]`` to embeddings ``[B, dim]``."""

    def __init__(self, dim: int, max_t: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim
        # None reproduces the unscaled sketch from the design note.
        self.max_t = max_t
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: ``[B]`` (or ``[B, 1]``) timestep indices -> ``[B, dim]``."""
        t = t.reshape(-1, 1).float()
        if self.max_t is not None:
            t = t / float(self.max_t)
        emb = self.mlp(t)
        assert emb.shape[-1] == self.dim
        return emb

    def extra_repr(self) -> str:
        return f"dim={self.dim}, max_t={self.max_t}"
