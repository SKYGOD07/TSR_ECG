"""Noise anomaly score (Stage 3).

The diffusion branch never reconstructs the ECG.  It compares the noise that
was actually injected by the forward process against the noise the predictor
recovered:

    A_t(x) = mean_{c, l} ( epsilon - epsilon_hat )^2

The predictor is trained on NORMAL ECGs only, so for a normal record
``epsilon_hat ~= epsilon`` and ``A_t`` is small; an abnormal record pushes the
encoder off its learned manifold and the mismatch grows.

Only the squared-error score is implemented here.  KL divergence between prior
and posterior noise is deliberately left out of the first experiment -- see
``docs/07_diffusion_noise_branch.md``.
"""

from typing import Optional

import torch


def noise_mse_score(epsilon: torch.Tensor, epsilon_hat: torch.Tensor) -> torch.Tensor:
    """Per-sample squared error between true and predicted noise.

    epsilon, epsilon_hat: ``[B, C, L]``
    returns: ``[B]`` -- one scalar per ECG, averaged over leads AND time.
    """
    if epsilon.shape != epsilon_hat.shape:
        raise ValueError(
            f"epsilon {tuple(epsilon.shape)} != epsilon_hat {tuple(epsilon_hat.shape)}"
        )
    if epsilon.dim() != 3:
        raise ValueError(f"expected [B, C, L], got shape {tuple(epsilon.shape)}")
    return (epsilon - epsilon_hat).pow(2).mean(dim=(1, 2))


@torch.no_grad()
def noise_score_at_t(
    model: torch.nn.Module,
    forward: "object",
    x0: torch.Tensor,
    t_public: int,
    repeats: int = 1,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Noise anomaly score for a batch at one PUBLIC timestep.

    model:     noise predictor, ``epsilon_hat = model(x_t, t_idx)`` (0-based t)
    forward:   :class:`diffusion.forward_process.ForwardDiffusion`
    x0:        ``[B, C, L]`` channel-first clean ECG
    repeats:   number of independent epsilon draws to average over.  A single
               draw makes the score noticeably noisy; averaging a few draws
               stabilises the resulting AUC without changing what is measured.

    returns: ``[B]`` scores.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    b = x0.shape[0]
    total = torch.zeros(b, device=x0.device, dtype=torch.float32)
    t_idx = None
    for _ in range(repeats):
        x_t, epsilon = forward.noise_at(x0, t_public, generator=generator)
        if t_idx is None:
            from .forward_process import t_public_to_index

            t_idx = t_public_to_index(t_public, forward.T, batch_size=b, device=x0.device)
        epsilon_hat = model(x_t, t_idx)
        assert epsilon_hat.shape == epsilon.shape, (
            f"predicted noise {tuple(epsilon_hat.shape)} must match "
            f"sampled noise {tuple(epsilon.shape)}"
        )
        total += noise_mse_score(epsilon, epsilon_hat)

    return total / repeats
