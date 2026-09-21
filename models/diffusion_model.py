"""TSR-based diffusion noise predictor (Stage 2).

Architecture (see the design note, section 8 -- reuse, do not rebuild)::

        x_t  [B, 12, L]
          |
          v
    Encoder1D                      <- the SAME class lib/TSRNet.py uses
          |
          v
        z_t  [B, Cz, Lz]
          +
    TimeEmbedding(t) [B, Cz]       <- broadcast over the Lz axis
          |
          v
    NoiseDecoder1D                 <- Decoder1D without the final Tanh
          |
          v
    epsilon_hat  [B, 12, L]

Two things are deliberate:

* The encoder is ``lib.modules.Encoder1D``, the same block TSR-Net's
  ``time_encoder`` is built from, so the diffusion branch inherits the TSR
  feature extractor rather than introducing a second architecture.  It is a
  separate INSTANCE with its own weights: the original TSR-Net checkpoints and
  the ``train.py`` / ``test.py`` behaviour are untouched.
* The head drops ``Tanh``.  TSR-Net reconstructs a signal normalised to
  [-1, 1]; predicted noise is an unbounded Gaussian, so a squashing
  nonlinearity would cap the achievable prediction.

The latent shape is PROBED at construction time instead of hard-coded, so the
head stays correct if the encoder or the window length changes.
"""

from typing import Tuple

import torch
import torch.nn as nn

from lib.modules import Encoder1D
from .time_embedding import TimeEmbedding


class NoiseDecoder1D(nn.Module):
    """Mirror of ``lib.modules.Decoder1D`` with no output nonlinearity.

    Maps ``[B, 50, Lz]`` back to ``[B, nc, Lz * 32 + ...]``; with Lz = 136 and
    nc = 12 that is exactly ``[B, 12, 4800]``.
    """

    def __init__(self, nc: int) -> None:
        super().__init__()
        ngf = 32
        self.main = nn.Sequential(
            nn.ConvTranspose1d(50, ngf * 16, 15, 1, 0),
            nn.BatchNorm1d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 16, ngf * 8, 4, 2, 1),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            # No Tanh: epsilon_hat must be free to take any real value.
            nn.ConvTranspose1d(ngf, nc, 4, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class DiffusionNoisePredictor(nn.Module):
    """``epsilon_hat = g(TSR_encoder(x_t), TimeEmbedding(t))``."""

    def __init__(self, channels: int = 12, signal_length: int = 4800, max_t: int = 100) -> None:
        super().__init__()
        self.channels = channels
        self.signal_length = signal_length

        # 1. Feature extraction -- the TSR-Net 1-D encoder block.
        self.encoder = Encoder1D(channels)

        # 2. Probe the real latent shape rather than assuming it.
        latent_channels, latent_length = self._probe_latent_shape(channels, signal_length)
        self.latent_channels = latent_channels
        self.latent_length = latent_length

        # 3. Timestep conditioning, one scalar per latent channel.
        self.time_embedding = TimeEmbedding(latent_channels, max_t=max_t)

        # 4. Noise-prediction head.
        self.decoder = NoiseDecoder1D(channels)

    @torch.no_grad()
    def _probe_latent_shape(self, channels: int, signal_length: int) -> Tuple[int, int]:
        was_training = self.encoder.training
        self.encoder.eval()  # eval mode: BatchNorm uses its (untouched) running stats
        dummy = torch.zeros(2, channels, signal_length)
        z = self.encoder(dummy)
        self.encoder.train(was_training)
        assert z.dim() == 3, f"encoder must emit [B, Cz, Lz], got {tuple(z.shape)}"
        return int(z.shape[1]), int(z.shape[2])

    def encode(self, x_t: torch.Tensor) -> torch.Tensor:
        """``[B, C, L]`` -> latent ``[B, Cz, Lz]``.  Exposed for step analysis."""
        return self.encoder(x_t)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t: ``[B, C, L]`` channel-first; t: ``[B]`` ZERO-based timesteps.

        returns ``epsilon_hat`` with exactly ``x_t.shape``.
        """
        if x_t.dim() != 3:
            raise ValueError(f"x_t must be [B, C, L], got shape {tuple(x_t.shape)}")
        if x_t.shape[1] != self.channels:
            raise ValueError(
                f"x_t has {x_t.shape[1]} channels, model was built for {self.channels}"
            )
        if t.shape[0] != x_t.shape[0]:
            raise ValueError(f"t has length {t.shape[0]}, expected B={x_t.shape[0]}")

        z_t = self.encoder(x_t)                       # [B, Cz, Lz]
        t_emb = self.time_embedding(t)                # [B, Cz]
        z_t = z_t + t_emb.unsqueeze(-1)               # broadcast over Lz
        epsilon_hat = self.decoder(z_t)               # [B, C, L]

        assert epsilon_hat.shape == x_t.shape, (
            f"epsilon_hat {tuple(epsilon_hat.shape)} must equal x_t {tuple(x_t.shape)}"
        )
        return epsilon_hat

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, signal_length={self.signal_length}, "
            f"latent=[{self.latent_channels}, {self.latent_length}]"
        )
