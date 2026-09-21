"""Shared plumbing for the exp0*.py scripts (paths, device, checkpoint loading).

Keeping this in one place means every experiment builds the diffusion model the
same way and reports the same provenance, so results across M1..M4 are
comparable by construction.
"""

import os
import random
import sys
from typing import Optional, Tuple

import numpy as np
import torch

# Make the repository root importable when a script is run from anywhere.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from diffusion.forward_process import ECG_WINDOW_LENGTH, ForwardDiffusion  # noqa: E402
from diffusion.schedule import DiffusionSchedule  # noqa: E402
from models.diffusion_model import DiffusionNoisePredictor  # noqa: E402

DEFAULT_DIFFUSION_CKPT = "ckpt/diffusion/DiffusionNet-best.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(gpu: str = "0", require_cuda: bool = False) -> torch.device:
    """Resolve the device and say so out loud -- never fall back silently."""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
        print(f"device: {device} ({torch.cuda.get_device_name(int(gpu))})")
        return device
    if require_cuda:
        raise RuntimeError("CUDA is required for this experiment but is not available.")
    print("device: cpu  (WARNING: no CUDA -- expect this to be slow)")
    return torch.device("cpu")


def build_diffusion(
    device: torch.device,
    T: int = 100,
    dims: int = 12,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    signal_length: int = ECG_WINDOW_LENGTH,
) -> Tuple[DiffusionNoisePredictor, ForwardDiffusion]:
    """Untrained predictor + forward process, both on ``device``."""
    schedule = DiffusionSchedule(T=T, beta_start=beta_start, beta_end=beta_end).to(device)
    model = DiffusionNoisePredictor(
        channels=dims, signal_length=signal_length, max_t=T
    ).to(device)
    return model, ForwardDiffusion(schedule)


def load_diffusion_checkpoint(
    model: DiffusionNoisePredictor,
    ckpt_path: str,
    device: torch.device,
    required: bool = True,
) -> Optional[dict]:
    """Load a diffusion checkpoint; raise (not warn) when it is required.

    Scoring with an untrained predictor produces numbers that look like results
    but mean nothing, so the evaluation experiments refuse to run without one.
    """
    if not os.path.exists(ckpt_path):
        message = (
            f"Diffusion checkpoint not found: {ckpt_path}\n"
            "Run Stage 2 training first:\n"
            "    python training/train_diffusion.py --epochs 10 --batch_size 32"
        )
        if required:
            raise FileNotFoundError(message)
        print(f"WARNING: {message}")
        return None

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    config = checkpoint.get("config", {})
    print(
        f"loaded {ckpt_path}  (epoch {checkpoint.get('epoch', '?')}, "
        f"val_mse={checkpoint.get('val_loss', float('nan')):.6f}, "
        f"T={config.get('T', '?')})"
    )
    return checkpoint


def require_data(data_path: str, *files: str) -> None:
    """Fail early and clearly if the preprocessed arrays are missing."""
    missing = [f for f in files if not os.path.exists(os.path.join(data_path, f))]
    if missing:
        raise FileNotFoundError(
            f"missing in {data_path}/: {', '.join(missing)}. "
            "Run preprocess.py (or the notebook preprocessing cell) first."
        )
