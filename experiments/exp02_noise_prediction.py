"""STAGE 2 SMOKE TEST -- one forward/backward pass of the noise predictor.

Runs a small batch all the way through

    x0 -> sample t -> sample epsilon -> x_t -> TSR encoder -> epsilon_hat -> MSE

and prints every tensor shape plus the loss.  This is the gate the design note
asks for: do not start the full training run until it passes.

An untrained predictor should land near MSE ~= 1.0, because the target is
standard Gaussian noise (E[eps^2] = 1) and an untrained head predicts roughly
zero.  A value far from that means the head is mis-scaled, not that training
has begun.

    python experiments/exp02_noise_prediction.py
    python experiments/exp02_noise_prediction.py --batch_size 16 --steps 3
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from _common import REPO_ROOT, build_diffusion, get_device, require_data, set_seed

from diffusion.forward_process import (
    ECG_LEADS,
    ECG_WINDOW_LENGTH,
    crop_window,
    per_lead_minmax,
    to_channel_first,
)


def main(args):
    set_seed(args.seed)
    os.chdir(REPO_ROOT)
    require_data(args.data_path, "train.npy")
    device = get_device(args.gpu)

    print("STAGE 2 -- diffusion noise-predictor smoke test")
    data = np.load(os.path.join(args.data_path, "train.npy"), mmap_mode="r")
    batch = crop_window(np.asarray(data[: args.batch_size]))       # (B, 4800, 12)
    if args.normalize:
        batch = np.stack([per_lead_minmax(rec) for rec in batch])

    # (B, 4800, 12) -> [B, 12, 4800]
    x0 = to_channel_first(torch.tensor(batch, dtype=torch.float32)).to(device)

    model, forward = build_diffusion(device, T=args.T, dims=args.dims)
    print(f"model: {model.extra_repr()}")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {round(n_parameters * 1e-6, 2)} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses = []

    for step in range(args.steps):
        t_idx = forward.schedule.sample_t(x0.shape[0])       # zero-based 0..T-1
        epsilon = torch.randn_like(x0)
        x_t, epsilon = forward.schedule.add_noise(x0, t_idx, noise=epsilon)
        epsilon_hat = model(x_t, t_idx)
        loss = nn.functional.mse_loss(epsilon_hat, epsilon)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step == 0:
            print()
            print(f"  x0.shape          {tuple(x0.shape)}")
            print(f"  xt.shape          {tuple(x_t.shape)}")
            print(f"  epsilon.shape     {tuple(epsilon.shape)}")
            print(f"  epsilon_hat.shape {tuple(epsilon_hat.shape)}")
            print(f"  t (public 1..{args.T}) {sorted((t_idx + 1).tolist())}")
            print(f"  loss.item()       {loss.item():.6f}")
            print()

            expected = (args.batch_size, ECG_LEADS, ECG_WINDOW_LENGTH)
            assert tuple(x0.shape) == expected, f"x0 {tuple(x0.shape)} != {expected}"
            assert x_t.shape == x0.shape, "x_t must match x0"
            assert epsilon.shape == x0.shape, "epsilon must match x0"
            assert epsilon_hat.shape == epsilon.shape, "epsilon_hat must match epsilon"
            assert torch.isfinite(epsilon_hat).all(), "epsilon_hat contains NaN or Inf"
            assert torch.isfinite(loss), "loss is NaN or Inf"

        print(f"  step {step + 1}/{args.steps}  mse={loss.item():.6f}")

    grad_norm = sum(
        p.grad.detach().pow(2).sum().item() for p in model.parameters() if p.grad is not None
    ) ** 0.5
    print(f"\ngradient L2 norm after the last step: {grad_norm:.4f}")
    assert grad_norm > 0, "no gradient reached the parameters -- the graph is broken"

    print("\nSTAGE 2 smoke test passed. Shapes align and gradients flow.")
    print("Next: python training/train_diffusion.py --epochs 10 --batch_size 32")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 -- noise-prediction smoke test")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3, help="optimizer steps to run")
    parser.add_argument("--dims", type=int, default=12)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=668)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--normalize", type=int, default=0)
    args = parser.parse_args()
    args.normalize = bool(args.normalize)
    main(args)
