"""STAGE 5 -- fuse the TSR anomaly score with the diffusion noise score.

    A_final = alpha * A_TSR + (1 - alpha) * A_noise

with a FIXED alpha (0.5 by default).  alpha is not learned and not adaptive:
the point of this stage is to establish whether a plain convex combination of
the two scores beats either alone, measured on the same test set.

Reads ``results/scores.csv`` written by ``exp04_tsr_vs_noise.py`` so that M1,
M2 and M4 are computed from identical per-sample scores.  The alpha sweep is
reported in full, and the headline fused number is the one at ``--alpha`` --
the best alpha in the sweep is labelled as tuned on the test set, because it is.

    python experiments/exp05_fusion.py
    python experiments/exp05_fusion.py --alpha 0.5
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from _common import REPO_ROOT, set_seed

from evaluation.anomaly_scores import fuse_scores
from evaluation.metrics import min_max_normalize, roc_auc, summarize


def main(args):
    set_seed(args.seed)
    os.chdir(REPO_ROOT)

    scores_csv = os.path.join(args.results_dir, "scores.csv")
    if not os.path.exists(scores_csv):
        raise FileNotFoundError(
            f"{scores_csv} not found. Run it first:\n"
            "    python experiments/exp04_tsr_vs_noise.py --spec 1"
        )
    frame = pd.read_csv(scores_csv)
    for column in ("label", "tsr_score", "noise_score"):
        if column not in frame.columns:
            raise ValueError(f"{scores_csv} is missing the {column!r} column")

    meta_path = os.path.join(args.results_dir, "scores_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    t_public = meta.get("t", "?")

    labels = frame["label"].to_numpy().astype(int)
    tsr = frame["tsr_score"].to_numpy()
    noise = frame["noise_score"].to_numpy()

    print("STAGE 5 -- TSR + diffusion noise fusion")
    print(f"source: {scores_csv}  ({len(frame)} records, "
          f"normal={int((labels == 0).sum())}, abnormal={int((labels == 1).sum())})")
    if meta:
        print(f"provenance: t={t_public}, repeats={meta.get('repeats')}, seed={meta.get('seed')}, "
              f"spec={meta.get('spec')}, mask_loss={meta.get('mask_loss')}, "
              f"normalize_eval={meta.get('normalize_eval')}")
        print(f"            TSR ckpt {meta.get('tsr_ckpt')} | "
              f"diffusion ckpt {meta.get('diffusion_ckpt')}")

    m1 = summarize(labels, tsr)
    m2 = summarize(labels, noise)
    fused = fuse_scores(tsr, noise, alpha=args.alpha)
    m4 = summarize(labels, fused)

    corr = float(np.corrcoef(min_max_normalize(tsr), min_max_normalize(noise))[0, 1])

    print("\n--- ablation (identical test set and protocol) ---")
    header = f"{'model':<44}{'ROC-AUC':>10}{'PR-AUC':>10}"
    print(header)
    print("-" * len(header))
    print(f"{'M1  TSR only':<44}{m1['roc_auc']:>10.4f}{m1['pr_auc']:>10.4f}")
    print(f"{f'M2  Noise only (t={t_public})':<44}{m2['roc_auc']:>10.4f}{m2['pr_auc']:>10.4f}")
    print(f"{f'M4  TSR + Noise (alpha={args.alpha})':<44}{m4['roc_auc']:>10.4f}{m4['pr_auc']:>10.4f}")

    best = max((m1["roc_auc"], "M1"), (m2["roc_auc"], "M2"), (m4["roc_auc"], "M4"))
    delta = m4["roc_auc"] - max(m1["roc_auc"], m2["roc_auc"])
    print(f"\ncorrelation between the two normalised scores: {corr:+.4f}")
    print(f"best of the three at alpha={args.alpha}: {best[1]} (ROC-AUC {best[0]:.4f})")
    print(f"fusion vs. the better single branch: {delta:+.4f} ROC-AUC")
    if delta <= 0:
        print("Fusion did NOT improve on the better single branch at this alpha. "
              "Reported as measured.")

    print("\n--- alpha sweep (alpha = weight on the TSR score) ---")
    sweep_rows = []
    for alpha in args.alphas:
        auc = roc_auc(labels, fuse_scores(tsr, noise, alpha=alpha))
        sweep_rows.append({"alpha": alpha, "roc_auc": auc})
        marker = "  <- reported" if abs(alpha - args.alpha) < 1e-9 else ""
        print(f"  alpha={alpha:>4.2f}  ROC-AUC={auc:.4f}{marker}")

    sweep = pd.DataFrame(sweep_rows)
    best_row = sweep.loc[sweep["roc_auc"].idxmax()]
    print(f"\nbest alpha in this sweep: {best_row['alpha']:.2f} "
          f"(ROC-AUC {best_row['roc_auc']:.4f}) -- NOTE: selected on the test set, "
          "so it is an upper bound, not a held-out result.")

    os.makedirs(args.results_dir, exist_ok=True)
    ablation = pd.DataFrame([
        {"model": "M1_tsr_only", **m1},
        {"model": f"M2_noise_only_t{t_public}", **m2},
        {"model": f"M4_fusion_alpha{args.alpha}", **m4},
    ])
    ablation_csv = os.path.join(args.results_dir, "fusion_ablation.csv")
    sweep_csv = os.path.join(args.results_dir, "fusion_alpha_sweep.csv")
    ablation.to_csv(ablation_csv, index=False)
    sweep.to_csv(sweep_csv, index=False)
    print(f"\nwrote {ablation_csv}")
    print(f"wrote {sweep_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5 -- fixed-alpha score fusion")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="weight on the TSR score; fixed, not learned")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    parser.add_argument("--seed", type=int, default=668)
    args = parser.parse_args()
    main(args)
