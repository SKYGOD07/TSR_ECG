"""Stage 2 training: the DDPM noise-prediction objective on NORMAL ECGs only.

    x_t, epsilon = forward_diffusion(x_0, t)
    epsilon_hat  = model(x_t, t)
    loss         = MSE(epsilon_hat, epsilon)

Nothing else: no KL, no adversarial, no contrastive, no reconstruction and no
anomaly-classification term.  Keeping the first experiment to the plain
noise-prediction objective is what makes the Stage 3 score interpretable.

Why normal-only
---------------
``data/train.npy`` already contains only records whose first PTB-XL diagnostic
superclass is NORM (see ``preprocess.preprocess_ptbxl``), so training on it
learns p(epsilon | normal ECG).  ``data/test.npy`` is never used for gradient
updates; a slice of the normal training set is held out for validation so that
"the loss decreased" can be checked on data the model did not fit.

Run:
    python training/train_diffusion.py --epochs 10 --batch_size 32
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Make the repository root importable when run as a script.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader import DiffusionECGSet  # noqa: E402
from diffusion.forward_process import ECG_LEADS, ECG_WINDOW_LENGTH, to_channel_first  # noqa: E402
from diffusion.schedule import DiffusionSchedule  # noqa: E402
from models.diffusion_model import DiffusionNoisePredictor  # noqa: E402


def build_loaders(args, use_cuda):
    """Split the normal-only training set into train / validation."""
    full = DiffusionECGSet(
        folder=args.data_path, split="train",
        normalize=args.normalize_train, limit=args.limit,
    )
    n_total = len(full)
    n_val = max(1, int(round(n_total * args.val_fraction)))
    n_train = n_total - n_val
    if n_train < 1:
        raise ValueError(f"val_fraction={args.val_fraction} leaves no training data (N={n_total})")

    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = torch.utils.data.random_split(full, [n_train, n_val], generator=generator)

    kwargs = {"num_workers": args.num_workers, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, **kwargs
    )
    return train_loader, val_loader, n_train, n_val


def run_epoch(model, schedule, loader, device, optimizer=None, desc=""):
    """One pass. ``optimizer=None`` -> evaluation (no grad, no update)."""
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for window, _ in tqdm(loader, desc=desc, leave=False):
            # [B, 4800, 12] -> [B, 12, 4800]; the stored layout is unchanged.
            x0 = to_channel_first(window.float().to(device))
            b = x0.shape[0]

            t = schedule.sample_t(b)                    # zero-based, uniform over 0..T-1
            epsilon = torch.randn_like(x0)
            x_t, epsilon = schedule.add_noise(x0, t, noise=epsilon)
            epsilon_hat = model(x_t, t)

            assert epsilon_hat.shape == epsilon.shape, (
                f"epsilon_hat {tuple(epsilon_hat.shape)} != epsilon {tuple(epsilon.shape)}"
            )
            loss = nn.functional.mse_loss(epsilon_hat, epsilon)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total += loss.item() * b
            count += b

    return total / max(count, 1)


def main(args):
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    print(f"device: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(int(args.gpu))}")
    else:
        print("WARNING: running on CPU. Full diffusion training on CPU is impractically slow.")
    print(f"args: {vars(args)}")

    train_loader, val_loader, n_train, n_val = build_loaders(args, use_cuda)
    print(f"normal-only training records: {n_train} train / {n_val} val (test.npy is NOT used)")

    schedule = DiffusionSchedule(
        T=args.T, beta_start=args.beta_start, beta_end=args.beta_end
    ).to(device)
    model = DiffusionNoisePredictor(
        channels=args.dims, signal_length=ECG_WINDOW_LENGTH, max_t=args.T
    ).to(device)

    print(f"schedule: {schedule.extra_repr()}  (public t = 1..{args.T})")
    print(f"model: {model.extra_repr()}")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {round(n_parameters * 1e-6, 2)} M")
    assert args.dims == ECG_LEADS or args.dims != 12, "dims should match the number of ECG leads"

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    os.makedirs(args.save_path, exist_ok=True)

    history = []
    best_val = float("inf")
    start = time.time()

    for epoch in range(args.epochs):
        train_loss = run_epoch(
            model, schedule, train_loader, device, optimizer,
            desc=f"train {epoch + 1}/{args.epochs}",
        )
        val_loss = run_epoch(
            model, schedule, val_loader, device, None,
            desc=f"val   {epoch + 1}/{args.epochs}",
        )
        elapsed = time.time() - start
        print(
            f"epoch {epoch + 1:>3}/{args.epochs}  "
            f"train_mse={train_loss:.6f}  val_mse={val_loss:.6f}  ({elapsed:.0f}s)"
        )
        history.append({"epoch": epoch + 1, "train_mse": train_loss, "val_mse": val_loss})

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": {
                "T": args.T,
                "beta_start": args.beta_start,
                "beta_end": args.beta_end,
                "dims": args.dims,
                "signal_length": ECG_WINDOW_LENGTH,
                "normalize_train": args.normalize_train,
                "seed": args.seed,
            },
        }
        torch.save(payload, os.path.join(args.save_path, "DiffusionNet-latest.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(payload, os.path.join(args.save_path, "DiffusionNet-best.pt"))
            print(f"  new best val_mse={val_loss:.6f} -> DiffusionNet-best.pt")

    with open(os.path.join(args.save_path, "train_history.json"), "w") as fh:
        json.dump({"args": vars(args), "history": history}, fh, indent=2)

    print(f"best val_mse: {best_val:.6f}")
    print(f"checkpoints: {os.path.join(args.save_path, 'DiffusionNet-{best,latest}.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2 -- train the TSR-based diffusion noise predictor on normal ECGs only"
    )
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dims", type=int, default=12, help="number of ECG leads")
    parser.add_argument("--save_path", type=str, default="ckpt/diffusion/",
                        help="kept separate from the TSR-Net checkpoints in ckpt/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=668)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--T", type=int, default=100, help="total diffusion timesteps")
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N normal records (smoke tests)")
    parser.add_argument("--val_fraction", type=float, default=0.05,
                        help="fraction of the normal training set held out for validation")
    parser.add_argument("--normalize_train", type=int, default=0,
                        help="re-apply per-lead min-max to train.npy; it is already normalised, "
                             "so the default 0 is a no-op")
    args = parser.parse_args()
    args.normalize_train = bool(args.normalize_train)
    main(args)
