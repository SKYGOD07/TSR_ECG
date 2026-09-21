"""Forward-diffusion façade: tensor orientation + timestep conventions (Stage 1).

Why this module exists
----------------------
Two conventions collide in this project and both must be handled explicitly
rather than by silent reshaping:

1. ORIENTATION.  The stored dataset is channel-LAST, ``[N, 5000, 12]``
   (N records, 5000 samples, 12 leads), and the TSR-Net dataloader yields
   windows as ``[B, 4800, 12]``.  The diffusion math and the TSR encoder
   (``lib.modules.Encoder1D``) are channel-FIRST, ``[B, 12, 4800]``.
   The on-disk format is NEVER changed; conversion happens here, in one place.

2. TIMESTEPS.  The research write-up talks about ``t = 1 .. T`` (1-based).
   ``diffusion.schedule.DiffusionSchedule`` is indexed ``0 .. T-1`` (0-based).
   ``ForwardDiffusion`` takes PUBLIC 1-based ``t`` and converts once, so no
   off-by-one can leak into the experiments.

Window convention
-----------------
``dataloader.TrainSet``/``TestSet`` crop ``[100:4900]`` off the 5000-sample
record before handing it to TSR-Net, giving 4800 samples -- which is what the
five stride-2 convolutions of ``Encoder1D`` expect (4800 / 32 = 150, then a
width-15 valid conv -> 136).  Diffusion uses the SAME window so that the TSR
encoder can be reused unchanged.
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .schedule import DiffusionSchedule

#: Crop applied to the stored 5000-sample record before it reaches TSR-Net.
ECG_WINDOW_START = 100
ECG_WINDOW_END = 4900
#: Resulting window length, i.e. ``L`` in ``[B, 12, L]``.
ECG_WINDOW_LENGTH = ECG_WINDOW_END - ECG_WINDOW_START  # 4800
#: Number of ECG leads.
ECG_LEADS = 12


def crop_window(x: np.ndarray) -> np.ndarray:
    """Crop the stored record(s) to the TSR-Net window along the TIME axis.

    Accepts ``[5000, 12]`` or ``[N, 5000, 12]`` and returns ``[4800, 12]`` /
    ``[N, 4800, 12]``.  The on-disk array is not modified.
    """
    if x.ndim == 2:
        return x[ECG_WINDOW_START:ECG_WINDOW_END, :]
    if x.ndim == 3:
        return x[:, ECG_WINDOW_START:ECG_WINDOW_END, :]
    raise ValueError(f"expected a 2-D or 3-D array, got shape {x.shape}")


def per_lead_minmax(x: np.ndarray) -> np.ndarray:
    """Scale each lead of ONE record to [-1, 1] (per-record, per-lead min-max).

    This is exactly the transform ``preprocess.normalize`` already baked into
    ``data/train.npy``.  ``data/test.npy`` was saved WITHOUT it (see
    ``preprocess.denoise_test``), so the diffusion evaluation path re-applies it
    to keep the model's input distribution identical at train and test time.
    It is unsupervised and computed per record, so it leaks no label information.

    x: ``[L, C]`` -> ``[L, C]`` float64.
    """
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D [L, C] record, got shape {x.shape}")
    out = np.array(x, dtype=np.float64, copy=True)
    for lead in range(out.shape[1]):
        seq = out[:, lead]
        lo, hi = seq.min(), seq.max()
        out[:, lead] = 2.0 * (seq - lo) / (hi - lo) - 1.0 if hi > lo else 0.0
    return out


def to_channel_first(x: torch.Tensor) -> torch.Tensor:
    """``[B, L, C]`` -> ``[B, C, L]``.  Explicit, never implicit."""
    if x.dim() != 3:
        raise ValueError(f"expected [B, L, C], got shape {tuple(x.shape)}")
    return x.transpose(1, 2).contiguous()


def to_channel_last(x: torch.Tensor) -> torch.Tensor:
    """``[B, C, L]`` -> ``[B, L, C]``.  Inverse of :func:`to_channel_first`."""
    if x.dim() != 3:
        raise ValueError(f"expected [B, C, L], got shape {tuple(x.shape)}")
    return x.transpose(1, 2).contiguous()


def t_public_to_index(
    t: Union[int, Sequence[int], torch.Tensor],
    T: int,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert PUBLIC 1-based timestep(s) to a 0-based ``[B]`` long tensor.

    ``t`` may be a scalar (broadcast to ``batch_size``) or a per-sample sequence.
    Raises if any value falls outside ``1 .. T``.
    """
    if isinstance(t, torch.Tensor):
        idx = t.to(dtype=torch.long)
    elif isinstance(t, (int, np.integer)):
        if batch_size is None:
            raise ValueError("batch_size is required when t is a scalar")
        idx = torch.full((batch_size,), int(t), dtype=torch.long)
    else:
        idx = torch.as_tensor(list(t), dtype=torch.long)

    if idx.dim() == 0:
        if batch_size is None:
            raise ValueError("batch_size is required when t is a 0-dim tensor")
        idx = idx.repeat(batch_size)
    if batch_size is not None and idx.shape[0] != batch_size:
        raise ValueError(f"t has length {idx.shape[0]}, expected {batch_size}")

    lo, hi = int(idx.min()), int(idx.max())
    if lo < 1 or hi > T:
        raise ValueError(f"public timesteps must lie in [1, {T}], got [{lo}, {hi}]")

    idx = idx - 1  # public 1..T -> internal 0..T-1
    return idx.to(device) if device is not None else idx


def t_index_to_public(t_idx: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`t_public_to_index`."""
    return t_idx + 1


class ForwardDiffusion:
    """Thin wrapper over :class:`DiffusionSchedule` speaking PUBLIC 1-based ``t``.

    Nothing here is trainable -- Stage 1 adds noise and measures it, no more.
    """

    def __init__(self, schedule: DiffusionSchedule) -> None:
        self.schedule = schedule

    @property
    def T(self) -> int:
        return self.schedule.T

    @property
    def device(self) -> torch.device:
        return self.schedule.device

    def noise_at(
        self,
        x0: torch.Tensor,
        t_public: Union[int, Sequence[int], torch.Tensor],
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply forward diffusion at PUBLIC timestep(s) ``t_public``.

        x0: ``[B, C, L]`` channel-first.
        returns ``(x_t, epsilon)``, both ``x0.shape``.
        """
        if x0.dim() != 3:
            raise ValueError(f"x0 must be [B, C, L], got shape {tuple(x0.shape)}")
        b = x0.shape[0]
        t_idx = t_public_to_index(t_public, self.T, batch_size=b, device=x0.device)
        if noise is None and generator is not None:
            noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
        return self.schedule.add_noise(x0, t_idx, noise=noise)

    def check(self, x0: torch.Tensor, x_t: torch.Tensor, epsilon: torch.Tensor) -> None:
        """Stage-1 acceptance assertions: shapes match and nothing is NaN/Inf."""
        assert x_t.shape == x0.shape, f"x_t {tuple(x_t.shape)} != x0 {tuple(x0.shape)}"
        assert epsilon.shape == x0.shape, f"epsilon {tuple(epsilon.shape)} != x0 {tuple(x0.shape)}"
        for name, tensor in (("x0", x0), ("x_t", x_t), ("epsilon", epsilon)):
            assert torch.isfinite(tensor).all(), f"{name} contains NaN or Inf"
