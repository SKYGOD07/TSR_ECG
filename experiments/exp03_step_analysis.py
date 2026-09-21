"""STAGE 3 + STAGE 4 -- noise anomaly score, and how it varies with t.

Stage 3: for every test ECG, at a given timestep,

    A_t = mean_{lead, time} ( epsilon - epsilon_hat )^2

one scalar per ECG per timestep.

Stage 4: sweep t over the requested grid and report, per timestep, the mean
normal score, the mean abnormal score, their difference, and ROC/PR-AUC.  The
best timestep is reported as the argmax of MEASURED AUC -- nothing here assumes
which region of the schedule is informative.

Outputs
    Images/step_analysis_scores.png
    Images/step_analysis_auc.png
    results/step_analysis_summary.csv     one row per t
    results/step_analysis_per_sample.csv  one row per (sample, t)

    python experiments/exp03_step_analysis.py
    python experiments/exp03_step_analysis.py --repeats 3 --limit 200
"""

import argparse
import os

import numpy as np
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

from dataloader import DiffusionECGSet
from evaluation.step_analysis import (
    DEFAULT_TIMESTEPS,
    analyse_timesteps,
    best_timestep,
    load_test_classes,
    plot_step_analysis,
)


def main(args):
    set_seed(args.seed)
    os.chdir(REPO_ROOT)
    require_data(args.data_path, "test.npy", "label.npy")
    device = get_device(args.gpu)

    print("STAGE 3/4 -- noise anomaly score across diffusion timesteps")

    dataset = DiffusionECGSet(
        folder=args.data_path, split="test", normalize=args.normalize_eval, limit=args.limit
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    labels = np.load(os.path.join(args.data_path, "label.npy")).astype(int)
    if args.limit is not None:
        labels = labels[: len(dataset)]

    classes = load_test_classes(args.data_path)
    if classes is not None:
        if args.limit is not None:
            classes = classes[: len(dataset)]
        print(f"test_class.npy found -- per-class breakdown enabled "
              f"({sorted(set(classes.tolist()))})")
    else:
        print("test_class.npy not found -- binary normal/abnormal analysis only. "
              "Class labels are not invented; rerun preprocess.py to produce it.")

    print(f"test records: {len(dataset)}  "
          f"(normal={int((labels == 0).sum())}, abnormal={int((labels == 1).sum())})")
    print(f"per-lead min-max applied to test windows: {args.normalize_eval}")
    if not args.normalize_eval:
        print("WARNING: test.npy was saved WITHOUT normalisation while train.npy was saved WITH "
              "it, so the model is being scored off its training amplitude distribution.")

    model, forward = build_diffusion(device, T=args.T, dims=args.dims)
    load_diffusion_checkpoint(model, args.ckpt, device, required=not args.allow_untrained)

    print(f"timesteps: {args.timesteps}   epsilon draws averaged per sample: {args.repeats}")
    per_sample, summary = analyse_timesteps(
        model, forward, loader, labels, device,
        timesteps=args.timesteps, repeats=args.repeats, seed=args.seed, classes=classes,
    )

    print("\n--- per-timestep summary ---")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    t_star, auc_star = best_timestep(summary, "roc_auc")
    t_diff, diff_star = best_timestep(summary, "diff")
    print(f"\nbest MEASURED timestep by ROC-AUC:    t={t_star} (AUC={auc_star:.4f})")
    print(f"best MEASURED timestep by score gap:  t={t_diff} (abnormal-normal={diff_star:+.6f})")
    print("These are the argmax over the evaluated grid only -- not a claim about "
          "untested timesteps.")

    os.makedirs(args.results_dir, exist_ok=True)
    summary_csv = os.path.join(args.results_dir, "step_analysis_summary.csv")
    per_sample_csv = os.path.join(args.results_dir, "step_analysis_per_sample.csv")
    summary.to_csv(summary_csv, index=False)
    per_sample.to_csv(per_sample_csv, index=False)
    print(f"\nwrote {summary_csv}")
    print(f"wrote {per_sample_csv}  ({len(per_sample)} rows)")

    for path in plot_step_analysis(summary, out_dir=args.out_dir):
        print(f"wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3/4 -- noise score and timestep analysis")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_DIFFUSION_CKPT)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dims", type=int, default=12)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--timesteps", type=int, nargs="+", default=list(DEFAULT_TIMESTEPS))
    parser.add_argument("--repeats", type=int, default=1,
                        help="epsilon draws averaged per sample; >1 reduces score variance")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N records")
    parser.add_argument("--seed", type=int, default=668)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--out_dir", type=str, default="Images")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--normalize_eval", type=int, default=1,
                        help="per-lead min-max on test windows so they match the train "
                             "distribution; see dataloader.DiffusionECGSet")
    parser.add_argument("--allow_untrained", action="store_true",
                        help="run without a checkpoint (diagnostics only -- results are meaningless)")
    args = parser.parse_args()
    args.normalize_eval = bool(args.normalize_eval)
    main(args)
