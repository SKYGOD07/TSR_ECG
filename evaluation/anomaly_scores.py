"""Anomaly scores for both branches, on a common per-sample footing.

Two scorers live here so that the M1/M2/M3/M4 ablations all read the SAME
numbers rather than each experiment re-deriving them:

* :func:`tsr_anomaly_scores`   -- the existing TSR-Net restoration error.  The
  loop is a faithful copy of ``test.py:detection_test`` (including rebuilding
  the mask for every ``j``), so it reproduces the published baseline instead of
  approximating it.
* :func:`noise_anomaly_scores` -- the diffusion prior/posterior noise mismatch.

Both return raw, UNNORMALISED scores indexed by position in the test set.
Normalisation and fusion are handled by :func:`fuse_scores`.
"""

import copy
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from diffusion.forward_process import ForwardDiffusion, to_channel_first, t_public_to_index
from diffusion.noise_score import noise_mse_score
from .metrics import min_max_normalize


# Kept under the original name so earlier scripts keep working.
def compute_noise_anomaly_score(epsilon: torch.Tensor, epsilon_hat: torch.Tensor) -> torch.Tensor:
    """``[B, C, L]`` pair -> ``[B]`` mean squared noise error.

    See :func:`diffusion.noise_score.noise_mse_score`.
    """
    return noise_mse_score(epsilon, epsilon_hat)


# ---------------------------------------------------------------------------
# Diffusion branch
# ---------------------------------------------------------------------------

@torch.no_grad()
def noise_anomaly_scores(
    model: torch.nn.Module,
    forward: ForwardDiffusion,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    t_public: int,
    repeats: int = 1,
    seed: Optional[int] = None,
    progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Score every sample in ``loader`` at one PUBLIC timestep ``t_public``.

    ``loader`` must yield ``(window, index)`` from
    :class:`dataloader.DiffusionECGSet` -- windows are ``[B, 4800, 12]``
    channel-last and are transposed here, explicitly.

    ``seed`` fixes the epsilon draws so a rerun reproduces the same AUC;
    ``repeats`` averages several draws to cut the sampling variance.

    returns ``(scores [N], indices [N])`` in loader order.
    """
    model.eval()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        # Decorrelate draws across timesteps while staying deterministic.
        generator.manual_seed(int(seed) + 1000 * int(t_public))

    scores: List[np.ndarray] = []
    indices: List[np.ndarray] = []
    iterator = tqdm(loader, desc=f"noise score t={t_public}", leave=False) if progress else loader

    for window, index in iterator:
        # [B, 4800, 12] -> [B, 12, 4800]; the stored layout is never mutated.
        x0 = to_channel_first(window.float().to(device))
        b = x0.shape[0]
        t_idx = t_public_to_index(t_public, forward.T, batch_size=b, device=device)

        total = torch.zeros(b, device=device, dtype=torch.float32)
        for _ in range(repeats):
            x_t, epsilon = forward.noise_at(x0, t_public, generator=generator)
            epsilon_hat = model(x_t, t_idx)
            assert epsilon_hat.shape == epsilon.shape
            total += noise_mse_score(epsilon, epsilon_hat)

        scores.append((total / repeats).cpu().numpy())
        indices.append(index.numpy())

    return np.concatenate(scores), np.concatenate(indices)


# ---------------------------------------------------------------------------
# TSR-Net branch
# ---------------------------------------------------------------------------

def load_tsr_model(ckpt_path: str, device: torch.device, dims: int = 12, spec: bool = False):
    """Load a TSR-Net checkpoint into the architecture that produced it.

    ``spec`` must match how the checkpoint was trained (``train.py --spec``):
    True -> ``lib.TSRNet.TSRNet`` (time + spectrogram),
    False -> ``lib.TSRNet_time.TSRNet_time`` (time only).
    A mismatch raises here rather than producing meaningless scores.
    """
    from lib.TSRNet import TSRNet
    from lib.TSRNet_time import TSRNet_time

    model = (TSRNet(enc_in=dims) if spec else TSRNet_time(enc_in=dims)).to(device)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"TSR-Net checkpoint not found: {ckpt_path}. Run the Stage 0 baseline first."
        )
    checkpoint = torch.load(ckpt_path, map_location=device)
    arch = "TSRNet" if spec else "TSRNet_time"
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as err:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} does not fit {arch}. "
            f"Set --spec to match how it was trained.\n{err}"
        ) from err
    model.eval()
    return model


@torch.no_grad()
def tsr_anomaly_scores(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    dims: int = 12,
    spec: bool = False,
    mask_loss: bool = False,
    mask_ratio_time: int = 30,
    mask_ratio_spec: int = 20,
    patch_length_div: int = 100,
    progress: bool = True,
) -> np.ndarray:
    """TSR-Net restoration-error score per sample -- same loop as ``test.py``.

    ``test_loader`` must be a :class:`dataloader.TestSet` loader with
    ``batch_size=1`` (the R-peak list has a per-sample length).
    """
    model.eval()
    result: List[float] = []
    iterator = tqdm(test_loader, desc="TSR score", leave=False) if progress else test_loader

    for time_ecg, spectrogram_ecg, r_index in iterator:
        if time_ecg.shape[0] != 1:
            raise ValueError("tsr_anomaly_scores requires batch_size=1 (variable-length r_index)")
        instance_result = []
        time_length = time_ecg.shape[1]

        loss_mask = None
        if mask_loss:
            # Peak-based error: only score a window around each detected R peak.
            idx_length = r_index.shape[1]
            loss_mask = torch.zeros((time_length, dims), dtype=torch.bool).to(device)
            for r_idx in range(idx_length):
                r_index_value = r_index[0][r_idx]
                if r_index_value > 200 and r_index_value < time_length - 400:
                    left = max(0, r_index_value - 240)
                    loss_mask[left:r_index_value + 240, :] = 1

        for j in range(patch_length_div // mask_ratio_time):
            patch_interval_time = time_length // mask_ratio_time
            time_ecg = time_ecg.float().to(device)
            mask_time = copy.deepcopy(time_ecg)
            # NOTE: the mask is rebuilt for every j -- each round masks
            # mask_ratio_time patches and the rounds are averaged. Accumulating
            # it across j would mask ~90% of the signal at once and would not
            # reproduce the baseline.
            mask = torch.zeros((1, time_length, 1), dtype=torch.bool).to(device)
            for k in range(mask_ratio_time):
                cut_idx = 48 * j + patch_interval_time * k
                mask[:, cut_idx:cut_idx + 48] = 1
            mask_time = torch.mul(mask_time, ~mask)

            if spec:
                patch_interval_spec = 66 // mask_ratio_spec
                spec_ecg = spectrogram_ecg.float().to(device)
                bs, freq_dim, time_dim, _ = spec_ecg.shape
                mask_spec = copy.deepcopy(spec_ecg)
                mask = torch.zeros((bs, freq_dim, time_dim, 1), dtype=torch.bool).to(device)
                for k in range(mask_ratio_spec):
                    cut_idx = 1 * j + patch_interval_spec * k
                    mask[:, :, cut_idx:cut_idx + 1] = 1
                mask_spec = torch.mul(mask_spec, ~mask)
                gen_time, time_var = model(mask_time, mask_spec)
            else:
                gen_time, time_var = model(mask_time)

            time_err = (gen_time - time_ecg) ** 2
            if mask_loss:
                l_time = torch.exp(-time_var) * time_err
                l_time = torch.mul(l_time, loss_mask)
                loss = torch.sum(l_time) / torch.sum(loss_mask)
            else:
                loss = torch.mean(torch.exp(-time_var) * time_err)

            instance_result.append(loss.detach().cpu().numpy())

        result.append(float(np.asarray(instance_result).mean()))

    return np.asarray(result)


# ---------------------------------------------------------------------------
# Fusion (Stage 5)
# ---------------------------------------------------------------------------

def fuse_scores(tsr_scores: np.ndarray, noise_scores: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """``A_final = alpha * A_TSR + (1 - alpha) * A_noise``.

    Both inputs are min-max normalised first: they live on entirely different
    scales (restoration error vs. noise MSE), so a raw sum would simply track
    whichever one happens to be larger.  ``alpha`` is fixed, not learned.
    """
    if tsr_scores.shape != noise_scores.shape:
        raise ValueError(
            "score vectors must align sample-by-sample: "
            f"{tsr_scores.shape} vs {noise_scores.shape}"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
    return alpha * min_max_normalize(tsr_scores) + (1.0 - alpha) * min_max_normalize(noise_scores)
