"""M1 vs M2 -- the TSR baseline against the diffusion noise score alone.

Computes both anomaly scores for the SAME test records, under the same split
and the same evaluation protocol, and writes them aligned sample-by-sample to
``results/scores.csv``.  ``exp05_fusion.py`` reads that file instead of
recomputing, so the fused numbers are guaranteed to come from these exact
scores.

    M1  TSR-Net restoration error            (existing baseline, test.py logic)
    M2  diffusion noise mismatch at t        (normal-only trained predictor)

``--spec`` must match how the TSR checkpoint was trained.  The notebook trains
the Stage 0 baseline with ``--spec True``, so pass ``--spec 1`` here to score it.

    python experiments/exp04_tsr_vs_noise.py --spec 1 --t 50
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from _common import (
    DEFAULT_DIFFUSION_CKPT,
    REPO_ROOT,
    build_diffusion,
    get_device,
    load_diffusion_checkpoint,
    require_data,
    set_seed,
)

from dataloader import DiffusionECGSet, TestSet
from evaluation.anomaly_scores import load_tsr_model, noise_anomaly_scores, tsr_anomaly_scores
from evaluation.metrics import summarize
from evaluation.step_analysis import load_test_classes


def pick_timestep(args) -> int:
    """Use --t when given, else the best MEASURED t from the Stage 4 summary.

    Falls back to the midpoint of the schedule only if neither exists, and says
    so -- an unmeasured default is never presented as an optimum.
    """
    if args.t is not None:
        print(f"timestep: t={args.t} (given on the command line)")
        return args.t

    summary_csv = os.path.join(args.results_dir, "step_analysis_summary.csv")
    if os.path.exists(summary_csv):
        summary = pd.read_csv(summary_csv).dropna(subset=["roc_auc"])
        if not summary.empty:
            row = summary.loc[summary["roc_auc"].idxmax()]
            t = int(row["t"])
            print(f"timestep: t={t} (best measured ROC-AUC={row['roc_auc']:.4f} "
                  f"from {summary_csv})")
            return t

    t = max(1, args.T // 2)
    print(f"timestep: t={t} (arbitrary midpoint -- no Stage 4 results found at {summary_csv}; "
          "run exp03_step_analysis.py to choose this empirically)")
    return t


def main(args):
    set_seed(args.seed)
    os.chdir(REPO_ROOT)
    require_data(args.data_path, "test.npy", "label.npy")
    device = get_device(args.gpu)

    labels = np.load(os.path.join(args.data_path, "label.npy")).astype(int)
    n_records = len(labels) if args.limit is None else min(args.limit, len(labels))
    labels = labels[:n_records]
    classes = load_test_classes(args.data_path)
    if classes is not None:
        classes = classes[:n_records]

    print(f"test records: {n_records} "
          f"(normal={int((labels == 0).sum())}, abnormal={int((labels == 1).sum())})")

    # ---- M1: TSR-Net baseline --------------------------------------------
    print(f"\n[M1] TSR-Net ({'time+spectrogram' if args.spec else 'time only'})")
    tsr_set = TestSet(folder=args.data_path, fs=args.fs, nperseg=args.nperseg)
    if args.limit is not None:
        tsr_set = torch.utils.data.Subset(tsr_set, range(n_records))
    tsr_loader = torch.utils.data.DataLoader(tsr_set, batch_size=1, shuffle=False)

    tsr_model = load_tsr_model(args.tsr_ckpt, device, dims=args.dims, spec=args.spec)
    print(f"loaded {args.tsr_ckpt}")
    tsr_scores = tsr_anomaly_scores(
        tsr_model, tsr_loader, device,
        dims=args.dims, spec=args.spec, mask_loss=args.mask_loss,
        mask_ratio_time=args.mask_ratio_time, mask_ratio_spec=args.mask_ratio_spec,
        patch_length_div=args.patch_length_div,
    )
    del tsr_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- M2: diffusion noise score ---------------------------------------
    t_public = pick_timestep(args)
    print(f"\n[M2] diffusion noise score at t={t_public}")
    diff_set = DiffusionECGSet(
        folder=args.data_path, split="test", normalize=args.normalize_eval, limit=n_records
    )
    diff_loader = torch.utils.data.DataLoader(
        diff_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    diff_model, forward = build_diffusion(device, T=args.T, dims=args.dims)
    load_diffusion_checkpoint(diff_model, args.ckpt, device, required=True)

    noise_scores, indices = noise_anomaly_scores(
        diff_model, forward, diff_loader, device,
        t_public=t_public, repeats=args.repeats, seed=args.seed,
    )
    order = np.argsort(indices)
    noise_scores = noise_scores[order]

    assert tsr_scores.shape == noise_scores.shape, (
        f"score vectors must align: TSR {tsr_scores.shape} vs noise {noise_scores.shape}"
    )

    # ---- report and persist ----------------------------------------------
    m1 = summarize(labels, tsr_scores)
    m2 = summarize(labels, noise_scores)

    print("\n--- M1 vs M2 (same test set, same records) ---")
    header = f"{'model':<34}{'ROC-AUC':>10}{'PR-AUC':>10}{'mean(norm)':>14}{'mean(abn)':>14}"
    print(header)
    print("-" * len(header))
    print(f"{'M1  TSR-Net only':<34}{m1['roc_auc']:>10.4f}{m1['pr_auc']:>10.4f}"
          f"{m1['mean_normal']:>14.6f}{m1['mean_abnormal']:>14.6f}")
    print(f"{f'M2  Noise only (t={t_public})':<34}{m2['roc_auc']:>10.4f}{m2['pr_auc']:>10.4f}"
          f"{m2['mean_normal']:>14.6f}{m2['mean_abnormal']:>14.6f}")

    frame = pd.DataFrame({
        "sample_id": np.arange(n_records),
        "label": labels,
        "tsr_score": tsr_scores,
        "noise_score": noise_scores,
    })
    if classes is not None:
        frame["class"] = classes
    frame.attrs["t"] = t_public

    os.makedirs(args.results_dir, exist_ok=True)
    out_csv = os.path.join(args.results_dir, "scores.csv")
    frame.to_csv(out_csv, index=False)

    meta = {
        "t": t_public, "repeats": args.repeats, "seed": args.seed,
        "spec": args.spec, "mask_loss": args.mask_loss,
        "normalize_eval": args.normalize_eval,
        "tsr_ckpt": args.tsr_ckpt, "diffusion_ckpt": args.ckpt,
        "n_records": n_records, "T": args.T,
    }
    meta_path = os.path.join(args.results_dir, "scores_meta.json")
    with open(meta_path, "w") as fh:
        import json
        json.dump(meta, fh, indent=2)

    print(f"\nwrote {out_csv} and {meta_path}")
    print("Next: python experiments/exp05_fusion.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M1 (TSR) vs M2 (diffusion noise)")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--tsr_ckpt", type=str, default="ckpt/TSRNet-latest.pt")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_DIFFUSION_CKPT)
    parser.add_argument("--spec", type=int, default=1,
                        help="1 if the TSR checkpoint was trained with --spec True")
    parser.add_argument("--mask_loss", type=int, default=1,
                        help="peak-based error, matching the notebook baseline")
    parser.add_argument("--dims", type=int, default=12)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--t", type=int, default=None,
                        help="public timestep; default = best measured t from exp03")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=668)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--normalize_eval", type=int, default=1)
    parser.add_argument("--fs", type=int, default=500)
    parser.add_argument("--nperseg", type=int, default=125)
    parser.add_argument("--mask_ratio_time", type=int, default=30)
    parser.add_argument("--mask_ratio_spec", type=int, default=20)
    parser.add_argument("--patch_length_div", type=int, default=100)
    args = parser.parse_args()
    args.spec = bool(args.spec)
    args.mask_loss = bool(args.mask_loss)
    args.normalize_eval = bool(args.normalize_eval)
    main(args)
