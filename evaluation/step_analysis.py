"""Diffusion-timestep analysis (Stage 4).

The research question this answers is:

    At which diffusion timesteps does the prior/posterior noise mismatch
    separate normal from abnormal ECGs?

It is answered by MEASURING, never by assumption -- :func:`analyse_timesteps`
evaluates every requested t and returns a tidy per-sample table plus a
per-timestep summary.  :func:`best_timestep` reports the argmax of a stated
criterion; no timestep is called optimal anywhere else in the codebase.

Class-conditional analysis (t* per anomaly type) is supported only when
``data/test_class.npy`` exists -- see ``preprocess.py``.  PTB-XL diagnostic
superclasses are NORM / MI / STTC / CD / HYP; labels are never invented for
classes the dataset does not carry.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from diffusion.forward_process import ForwardDiffusion
from .anomaly_scores import noise_anomaly_scores
from .metrics import pr_auc, roc_auc

#: The timestep grid the design note asks for as a starting point.
DEFAULT_TIMESTEPS: Tuple[int, ...] = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


def analyse_timesteps(
    model: torch.nn.Module,
    forward: ForwardDiffusion,
    loader: torch.utils.data.DataLoader,
    labels: np.ndarray,
    device: torch.device,
    timesteps: Sequence[int] = DEFAULT_TIMESTEPS,
    repeats: int = 1,
    seed: Optional[int] = None,
    classes: Optional[np.ndarray] = None,
    progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the noise anomaly score at each timestep.

    labels:  ``[N]`` binary, 0 = normal, 1 = abnormal, aligned with the dataset.
    classes: optional ``[N]`` of PTB-XL superclass strings for the per-class
             breakdown.  Omitted when ``data/test_class.npy`` is absent.

    returns ``(per_sample, summary)``:

    * ``per_sample`` -- one row per (sample, t): sample_id, label, class, t,
      noise_score.  This is the raw evidence; nothing downstream re-derives it.
    * ``summary``    -- one row per t: mean_normal, mean_abnormal, diff,
      roc_auc, pr_auc, plus ``mean_<CLASS>`` / ``diff_<CLASS>`` when available.
    """
    labels = np.asarray(labels).astype(int)
    rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for t_public in timesteps:
        scores, indices = noise_anomaly_scores(
            model, forward, loader, device,
            t_public=t_public, repeats=repeats, seed=seed, progress=progress,
        )
        order = np.argsort(indices)
        indices = indices[order]
        scores = scores[order]
        y = labels[indices]

        for sid, lbl, score in zip(indices, y, scores):
            row = {"sample_id": int(sid), "label": int(lbl), "t": int(t_public),
                   "noise_score": float(score)}
            if classes is not None:
                row["class"] = str(classes[sid])
            rows.append(row)

        normal = scores[y == 0]
        abnormal = scores[y == 1]
        entry = {
            "t": int(t_public),
            "mean_normal": float(normal.mean()) if normal.size else float("nan"),
            "mean_abnormal": float(abnormal.mean()) if abnormal.size else float("nan"),
            "roc_auc": roc_auc(y, scores),
            "pr_auc": pr_auc(y, scores),
        }
        entry["diff"] = entry["mean_abnormal"] - entry["mean_normal"]

        if classes is not None:
            cls = np.asarray(classes)[indices]
            for name in sorted(set(cls.tolist())):
                sel = cls == name
                if not sel.any():
                    continue
                entry[f"mean_{name}"] = float(scores[sel].mean())
                if normal.size:
                    entry[f"diff_{name}"] = float(scores[sel].mean() - normal.mean())

        summary_rows.append(entry)
        print(
            f"  t={t_public:>3}  normal={entry['mean_normal']:.6f}  "
            f"abnormal={entry['mean_abnormal']:.6f}  diff={entry['diff']:+.6f}  "
            f"AUC={entry['roc_auc']:.4f}"
        )

    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def best_timestep(summary: pd.DataFrame, criterion: str = "roc_auc") -> Tuple[int, float]:
    """Return ``(t, value)`` maximising ``criterion`` over the MEASURED grid.

    The result is only ever the best of the timesteps that were actually
    evaluated -- it is not a claim about the continuum between them.
    """
    if criterion not in summary.columns:
        raise ValueError(f"criterion {criterion!r} not in summary columns {list(summary.columns)}")
    valid = summary.dropna(subset=[criterion])
    if valid.empty:
        raise ValueError(f"no usable values for criterion {criterion!r}")
    row = valid.loc[valid[criterion].idxmax()]
    return int(row["t"]), float(row[criterion])


def load_test_classes(data_path: str) -> Optional[np.ndarray]:
    """Load ``data/test_class.npy`` (PTB-XL superclass per test record), or None.

    Returns None when the file is absent -- in that case only the binary
    normal/abnormal analysis is available.  Class labels are never fabricated.
    """
    path = os.path.join(data_path, "test_class.npy")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def plot_step_analysis(summary: pd.DataFrame, out_dir: str = "Images") -> List[str]:
    """Write the two Stage-4 figures; returns the paths written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    written = []

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(summary["t"], summary["mean_normal"], marker="o", label="Normal")
    ax.plot(summary["t"], summary["mean_abnormal"], marker="o", label="Abnormal")
    ax.set_xlabel("Diffusion timestep t")
    ax.set_ylabel(r"Mean noise anomaly score  $A_t=\|\epsilon-\hat{\epsilon}\|^2$")
    ax.set_title("Noise anomaly score vs. diffusion timestep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "step_analysis_scores.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(summary["t"], summary["roc_auc"], marker="x", color="green", label="ROC-AUC")
    if "pr_auc" in summary:
        ax.plot(summary["t"], summary["pr_auc"], marker="+", color="purple", label="PR-AUC")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="chance (ROC)")
    ax.set_xlabel("Diffusion timestep t")
    ax.set_ylabel("AUC")
    ax.set_title("Detection performance vs. diffusion timestep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "step_analysis_auc.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path)

    return written
