"""Diffusion branch for the TSR-Net ECG anomaly detector.

Stage 1 (forward process) lives in :mod:`diffusion.schedule` and
:mod:`diffusion.forward_process`; the Stage 3 anomaly score lives in
:mod:`diffusion.noise_score`.  Nothing in this package trains anything -- the
noise predictor itself is in :mod:`models.diffusion_model`.
"""

from .schedule import DiffusionSchedule, extract
from .forward_process import (
    ECG_LEADS,
    ECG_WINDOW_END,
    ECG_WINDOW_LENGTH,
    ECG_WINDOW_START,
    ForwardDiffusion,
    crop_window,
    per_lead_minmax,
    t_index_to_public,
    t_public_to_index,
    to_channel_first,
    to_channel_last,
)
from .noise_score import noise_mse_score, noise_score_at_t

__all__ = [
    "DiffusionSchedule",
    "extract",
    "ForwardDiffusion",
    "crop_window",
    "per_lead_minmax",
    "to_channel_first",
    "to_channel_last",
    "t_public_to_index",
    "t_index_to_public",
    "ECG_LEADS",
    "ECG_WINDOW_START",
    "ECG_WINDOW_END",
    "ECG_WINDOW_LENGTH",
    "noise_mse_score",
    "noise_score_at_t",
]
