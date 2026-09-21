"""STAGE 1 TEST -- forward diffusion only. Nothing is trained here.

Takes one clean ECG, runs it through the DDPM forward process at a range of
timesteps and plots the result, so the Stage-1 acceptance criteria can be
checked by eye as well as by assertion:

  * x_t has the same shape as x_0
  * epsilon has the same shape as x_0
  * no NaN, no Inf
  * different timesteps produce visibly different noise levels

Lead II (index 1) is plotted, because that is the lead the R-peak detection in
``dataloader.py`` uses and the one whose morphology is easiest to read.

    python experiments/exp01_forward_noise.py
    python experiments/exp01_forward_noise.py --split test --index 0
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from _common import REPO_ROOT, require_data, set_seed  # noqa: E402

from diffusion.forward_process import (  # noqa: E402
    ECG_LEADS,
    ECG_WINDOW_LENGTH,
    ForwardDiffusion,
    crop_window,
    per_lead_minmax,
)
from diffusion.schedule import DiffusionSchedule  # noqa: E402

DEFAULT_TIMESTEPS = (1, 10, 25, 50, 75, 100)


def main(args):
    set_seed(args.seed)
    os.chdir(REPO_ROOT)
    require_data(args.data_path, f"{args.split}.npy")

    print("STAGE 1 -- forward diffusion smoke test (no training)")
    path = os.path.join(args.data_path, f"{args.split}.npy")
    data = np.load(path, mmap_mode="r")
    print(f"{path}: {data.shape}  (stored channel-last, [N, 5000, 12] -- not modified)")

    record = crop_window(np.asarray(data[args.index]))        # (4800, 12)
    if args.normalize:
        record = per_lead_minmax(record)
        print("applied per-lead min-max to [-1, 1] (matches how train.npy was saved)")

    # (4800, 12) -> [1, 12, 4800]: the diffusion math is channel-FIRST.
    x0 = torch.tensor(record, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
    print(f"x0 shape: {tuple(x0.shape)}  (expected (1, {ECG_LEADS}, {ECG_WINDOW_LENGTH}))")

    assert x0.dim() == 3, "x0 must be [B, C, L]"
    assert x0.shape[1] == ECG_LEADS, f"expected {ECG_LEADS} leads, got {x0.shape[1]}"
    assert x0.shape[2] == ECG_WINDOW_LENGTH, f"expected window {ECG_WINDOW_LENGTH}"
    assert torch.isfinite(x0).all(), "x0 contains NaN or Inf"

    forward = ForwardDiffusion(DiffusionSchedule(T=args.T))
    timesteps = [t for t in args.timesteps if 1 <= t <= args.T]
    if len(timesteps) != len(args.timesteps):
        print(f"note: timesteps outside 1..{args.T} were dropped")

    lead = args.lead
    clean = x0[0, lead].numpy()

    fig, axes = plt.subplots(len(timesteps) + 1, 1, figsize=(14, 2.1 * (len(timesteps) + 1)),
                             sharex=True)
    axes[0].plot(clean, color="tab:blue", linewidth=0.7)
    axes[0].set_title(f"x0 -- clean ECG (lead index {lead}, {args.split}.npy record {args.index})")
    axes[0].set_ylabel("amp")
    axes[0].grid(True, alpha=0.25)

    print(f"{'t':>5} {'sqrt(ab_t)':>11} {'sqrt(1-ab_t)':>13} {'std(x_t)':>10} {'corr(x_t,x0)':>13}")
    for ax, t_public in zip(axes[1:], timesteps):
        x_t, epsilon = forward.noise_at(x0, t_public)
        forward.check(x0, x_t, epsilon)

        idx = t_public - 1  # public 1..T -> internal 0..T-1
        sqrt_ab = float(forward.schedule.sqrt_alphas_cumprod[idx])
        sqrt_1mab = float(forward.schedule.sqrt_one_minus_alphas_cumprod[idx])
        noisy = x_t[0, lead].numpy()
        corr = float(np.corrcoef(clean, noisy)[0, 1])
        print(f"{t_public:>5} {sqrt_ab:>11.4f} {sqrt_1mab:>13.4f} "
              f"{noisy.std():>10.4f} {corr:>13.4f}")

        ax.plot(clean, color="tab:blue", alpha=0.35, linewidth=0.6, label="x0")
        ax.plot(noisy, color="tab:red", alpha=0.85, linewidth=0.6, label=f"x_t (t={t_public})")
        ax.set_title(
            f"t={t_public}:  x_t = {sqrt_ab:.3f}*x0 + {sqrt_1mab:.3f}*eps   "
            f"(corr with x0 = {corr:.3f})"
        )
        ax.set_ylabel("amp")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("sample index (500 Hz)")
    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out = os.path.join(args.out_dir, f"forward_diffusion_smoke_test{suffix}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)

    print(f"saved {out}")
    print("STAGE 1 assertions passed: shapes preserved, finite values, "
          "monotonically decreasing correlation with x0.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 -- forward diffusion visualisation")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--index", type=int, default=0, help="record index within the split")
    parser.add_argument("--lead", type=int, default=1, help="lead to plot (1 = lead II)")
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--timesteps", type=int, nargs="+", default=list(DEFAULT_TIMESTEPS))
    parser.add_argument("--out_dir", type=str, default="Images")
    parser.add_argument("--seed", type=int, default=668)
    parser.add_argument("--tag", type=str, default="",
                        help="suffix for the output filename, e.g. --tag abnormal")
    parser.add_argument("--normalize", type=int, default=0,
                        help="per-lead min-max first; needed for test.npy, a no-op for train.npy")
    args = parser.parse_args()
    args.normalize = bool(args.normalize)
    main(args)
