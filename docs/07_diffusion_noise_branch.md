# The Diffusion / Noise Branch

This document covers the diffusion extension added on top of the existing TSR-Net
anomaly detector: why it exists, the exact maths, how the code is laid out, how to
run it, and what it does **not** yet do.

If you have not read [04_the_tsrnet_architecture.md](04_the_tsrnet_architecture.md)
yet, start there — everything below builds on the TSR-Net encoder.

---

## 1. The existing TSR-Net baseline (Stage 0)

Nothing about it changed. `train.py`, `test.py`, `preprocess.py`, `dataloader.py`'s
`TrainSet`/`TestSet` and the `lib/` models all behave exactly as before.

```
PTB-XL
  └─► preprocess.py  (HeartPy filtering, per-lead normalisation, fold-10 split)
        └─► data/train.npy   normal-only, (N, 5000, 12)
            data/test.npy    normal + abnormal, (2155, 5000, 12)
            data/label.npy   0 = normal, 1 = abnormal
              └─► TSR-Net  (mask a patch, restore it)
                    └─► A_TSR = restoration error  ──►  ROC-AUC
```

TSR-Net is **restoration-based**: it masks patches of the ECG and measures how badly
it reconstructs them. Trained only on normal ECGs, it reconstructs normal records well
and abnormal records badly, and that error is the anomaly score.

Run it and write the number down. Every diffusion result is compared against it.

---

## 2. Why add diffusion

Restoration error is one view of "this record does not look like the normal data I was
trained on". Diffusion gives a second, quite different view.

Instead of reconstructing the ECG, the diffusion branch **corrupts it with a known
amount of Gaussian noise and asks the network to name the noise it added.** If the
network has learned the structure of normal ECGs, it can separate signal from noise on
a normal record. On an abnormal record, the unfamiliar morphology gets mistaken for
noise (or vice versa) and the prediction degrades.

The score is then the mismatch between the noise that was actually injected and the
noise that was recovered — a *prior vs. posterior noise* comparison rather than a
reconstruction comparison.

### ⚠️ HeartPy filtering is NOT the diffusion noise

This is the single easiest thing to confuse, so it is worth stating plainly:

| | HeartPy filtering (`preprocess.py`) | Diffusion noise (`diffusion/`) |
|---|---|---|
| **When** | before anything else | after preprocessing, at train/eval time |
| **Purpose** | *remove* baseline wander and mains hum | *add* a known, controlled corruption |
| **Written to disk?** | yes, baked into the `.npy` files | no, sampled fresh every batch |
| **Known to the model?** | no | yes — it is the training target |

Diffusion begins **after** the preprocessed arrays exist. It never runs before the
filtering/normalisation step, and it never modifies the stored arrays.

---

## 3. Stage 1 — forward diffusion

Standard DDPM. With `T = 100`, `beta_start = 1e-4`, `beta_end = 0.02`:

```
beta_t       linear from beta_start to beta_end
alpha_t      = 1 - beta_t
alpha_bar_t  = prod_{s<=t} alpha_s
```

and the closed-form forward step:

> **x_t = √(ᾱ_t) · x₀  +  √(1 − ᾱ_t) · ε**,  where  **ε ~ N(0, I)**

At small `t` the ECG is barely touched; at large `t` it is mostly noise. Measured on a
real record (`experiments/exp01_forward_noise.py`):

| t | √ᾱ_t | √(1−ᾱ_t) | correlation of x_t with x₀ |
|---:|---:|---:|---:|
| 1 | 0.9999 | 0.0100 | 0.999 |
| 10 | 0.9950 | 0.1000 | 0.952 |
| 25 | 0.9690 | 0.2469 | 0.767 |
| 50 | 0.8816 | 0.4720 | 0.493 |
| 75 | 0.7527 | 0.6584 | 0.343 |
| 100 | 0.6030 | 0.7978 | 0.245 |

`Images/forward_diffusion_smoke_test.png` shows the same thing visually: the QRS
complexes are clear through `t ≈ 10`, blur around `t ≈ 25–50`, and are buried by
`t ≈ 75–100`.

Stage 1 trains nothing.

### Two conventions, handled explicitly

**Orientation.** The arrays are channel-*last* on disk, `[N, 5000, 12]`. The diffusion
maths and `Encoder1D` are channel-*first*, `[B, 12, L]`. The stored format is never
changed — `diffusion/forward_process.py` does the conversion in one place, via
`to_channel_first` / `to_channel_last`. There is no silent reshaping anywhere.

**Window.** `dataloader.py` crops `[100:4900]` before the signal reaches TSR-Net, so
the working length is **4800**, not 5000. That is what `Encoder1D`'s five stride-2
convolutions expect (4800 / 32 = 150, then a width-15 valid conv → 136). Diffusion
uses the same window so the encoder can be reused unchanged.

**Timesteps.** The write-up talks about `t = 1..T`; the schedule tensors are indexed
`0..T-1`. `ForwardDiffusion` takes **public 1-based `t`** and converts once, so no
off-by-one can leak into an experiment. `DiffusionSchedule` is 0-based internally and
documented as such.

---

## 4. Stage 2 — the noise predictor

**Reuse, do not rebuild.** The predictor is built from the same `Encoder1D` block that
`lib/TSRNet.py` uses for its `time_encoder`:

```
x_t  [B, 12, 4800]
  │
  ▼
Encoder1D                          ← same class as TSR-Net's time encoder
  │
  ▼
z_t  [B, 50, 136]
  +
TimeEmbedding(t)  [B, 50]          ← broadcast across the 136 latent positions
  │
  ▼
NoiseDecoder1D                     ← Decoder1D minus the final Tanh
  │
  ▼
ε̂  [B, 12, 4800]
```

That is `ε̂ = g(TSR_encoder(x_t), TimeEmbedding(t))` — 2.17 M trainable parameters.

Three implementation notes:

* It is a **separate instance** with its own weights. The TSR-Net checkpoints in
  `ckpt/` and the behaviour of `train.py`/`test.py` are untouched.
* **The `Tanh` is removed.** TSR-Net reconstructs a signal normalised to [-1, 1], so a
  squashing output makes sense there. Predicted noise is an unbounded Gaussian; a
  `Tanh` would cap what the head can express.
* **The latent shape is probed, not hard-coded.** `DiffusionNoisePredictor.__init__`
  runs a dummy tensor through the encoder to discover `[50, 136]`, so the head stays
  correct if the encoder or the window length changes.

### Timestep embedding

```python
Linear(1, dim) → SiLU → Linear(dim, dim)
```

Deliberately the simple MLP, not a sinusoidal embedding. Its only job is to tell the
encoder which diffusion step produced `x_t`. The one addition is that `t` is divided by
`T` before entering the first `Linear`: a raw index up to 100 is a very large input
scale relative to the layer's initialisation. That is input scaling, not a different
embedding family.

### Training objective

```python
x_t, epsilon = schedule.add_noise(x0, t)
epsilon_hat  = model(x_t, t)
loss         = F.mse_loss(epsilon_hat, epsilon)
```

**Nothing else.** No KL, no adversarial, no contrastive, no reconstruction, no
anomaly-classification term. Keeping the first experiment to the plain noise-prediction
objective is what makes the Stage 3 score interpretable.

### Normal-only training

| split | contents | used for |
|---|---|---|
| `data/train.npy` (95%) | normal only | gradient updates |
| `data/train.npy` (5%) | normal only | validation |
| `data/test.npy` | normal + abnormal | **evaluation only** |

`data/train.npy` already contains only records whose first PTB-XL superclass is `NORM`
(see `preprocess.preprocess_ptbxl`), so training on it learns `p(ε | normal ECG)`.
`data/test.npy` is never used for gradient updates. The 5% held-out normal slice
exists so that "the loss went down" can be checked on data the model did not fit.

An untrained predictor sits at MSE ≈ 1.0, because the target is standard Gaussian noise
(E[ε²] = 1) and an untrained head outputs roughly zero. That is the number training has
to beat.

---

## 5. Stage 3 — the noise anomaly score

For each ECG, at each timestep, one scalar:

> **A_t = mean over leads and time of (ε − ε̂_t)²**

```python
noise_error = (epsilon - epsilon_pred).pow(2).mean(dim=(1, 2))   # -> [B]
```

MSE only. **KL divergence is deliberately not part of this first experiment** — it is
a candidate for later, once the MSE version is understood.

Because ε is sampled randomly, a single draw makes the score noisy. `--repeats N`
averages N independent draws per sample, and `--seed` fixes them, so a rerun
reproduces the same AUC.

Expected behaviour: for a normal ECG `ε̂ ≈ ε` and `A_t` is small; for an abnormal ECG
the mismatch is larger. **Whether that actually happens, and by how much, is what
Stage 4 measures — it is not assumed anywhere in the code.**

---

## 6. Stage 4 — which timesteps are informative

Sweep `t ∈ [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]` and report, per timestep:
mean normal score, mean abnormal score, their difference, ROC-AUC and PR-AUC.

Outputs:

| file | contents |
|---|---|
| `results/step_analysis_per_sample.csv` | one row per (sample_id, label, class, t, noise_score) |
| `results/step_analysis_summary.csv` | one row per t |
| `Images/step_analysis_scores.png` | mean normal vs. abnormal against t |
| `Images/step_analysis_auc.png` | ROC/PR-AUC against t |

`best_timestep()` reports the argmax over **the grid that was actually evaluated**, and
says so. No timestep is called optimal anywhere else in the codebase.

### Anomaly-specific timesteps

The research question is whether `t*` depends on the *kind* of abnormality:

> t*_c = argmax_t [ A_t(class c) − A_t(normal) ]

`data/label.npy` is binary and cannot answer this. `preprocess.py` therefore **also**
saves `data/test_class.npy` with the PTB-XL diagnostic superclass per test record —
`NORM`, `MI`, `STTC`, `CD`, `HYP` — in the same row order, keeping `label.npy`
byte-for-byte compatible.

Note what that does and does not give you. PTB-XL superclasses are **diagnostic
statement groups**, not rhythm labels. The design note mentions AF, PVC and ST
abnormality; only ST-related findings map cleanly here (`STTC`). **AF and PVC labels do
not exist in this data and are never fabricated.** Analysing them would require a
different label source or the PTB-XL sub-class/rhythm annotations.

When `test_class.npy` is absent the analysis runs binary-only and says so.

### Progression — do not skip ahead

The design note lays out five versions of timestep selection. Only the first two are
implemented:

| version | selection | status |
|---|---|---|
| 1 | fixed K | ✅ implemented (`--t`) |
| 2 | global best K | ✅ implemented (argmax of measured AUC) |
| 3 | anomaly-specific K_c | ⚠️ reported in the table; not used for scoring |
| 4 | sample-specific K_i = f(z_i) | ❌ not implemented |
| 5 | learnable weights w = f(z), A_N = Σ w_t A_t | ❌ not implemented |

Jumping straight to a learned step selector would make any gain impossible to
attribute. Versions 4–5 come after the fixed-K numbers are in hand.

---

## 7. Stage 5 — fusion

> **A_final = α · A_TSR + (1 − α) · A_noise**

with a **fixed** α = 0.5. Not learned, not adaptive.

The two scores live on completely different scales — TSR restoration error is ~0.2,
noise MSE is ~1.0 — so each is min-max normalised before combining. Min-max is
monotone, so it does not change either branch's own ROC-AUC; it exists purely to make
the sum meaningful.

`exp04_tsr_vs_noise.py` writes `results/scores.csv` with both scores aligned
sample-by-sample, and `exp05_fusion.py` reads that file. So M1, M2 and M4 are
guaranteed to come from identical per-sample scores on identical test records.

### Ablation

| | TSR | Diffusion | Noise score | Adaptive steps | status |
|---|---|---|---|---|---|
| **M1** | ✓ | – | – | – | ✅ |
| **M2** | – | ✓ | ✓ | – | ✅ |
| **M4** | ✓ | ✓ | ✓ | – | ✅ |
| **M5** | ✓ | ✓ | ✓ | global | ✅ (via `--t` from Stage 4) |
| **M6** | ✓ | ✓ | ✓ | anomaly-specific | ❌ |
| **M7** | ✓ | ✓ | ✓ | learnable | ❌ |

The α sweep is printed in full. The best α in the sweep is **selected on the test set**,
so it is an upper bound, not a held-out result — the script labels it as such. Report
the fixed α = 0.5 number as the headline. If fusion does not beat the better single
branch, the script says so; nothing is tuned to flatter the diffusion branch.

---

## 8. File layout

New files:

```
diffusion/
    __init__.py
    schedule.py          DDPM coefficients + closed-form add_noise (0-based t)
    forward_process.py   orientation, window, 1-based t, per-lead min-max
    noise_score.py       A_t = mean((eps - eps_hat)^2)

models/
    __init__.py
    time_embedding.py    Linear -> SiLU -> Linear
    diffusion_model.py   Encoder1D + time embedding + NoiseDecoder1D

training/
    train_diffusion.py   Stage 2, normal-only, held-out normal validation

evaluation/
    metrics.py           ROC-AUC, PR-AUC, min-max normalisation
    anomaly_scores.py    TSR scorer (matches test.py exactly) + noise scorer + fusion
    step_analysis.py     Stage 4 sweep, best_timestep, plots

experiments/
    _common.py           device, seeding, checkpoint loading
    exp01_forward_noise.py     Stage 1 visualisation
    exp02_noise_prediction.py  Stage 2 smoke test
    exp03_step_analysis.py     Stage 3 + 4
    exp04_tsr_vs_noise.py      M1 vs M2, writes results/scores.csv
    exp05_fusion.py            M4 + alpha sweep
```

Modified:

* `dataloader.py` — **added** `DiffusionECGSet` alongside the untouched
  `TrainSet`/`TestSet`. The originals compute an STFT spectrogram and a HeartPy R-peak
  list that the diffusion branch does not need; `TestSet`'s variable-length `r_index`
  also cannot be collated with a batch size > 1.
* `preprocess.py` — **additionally** saves `data/test_class.npy`. `train.npy`,
  `test.npy` and `label.npy` are byte-for-byte what they were.
* `TSRNet_Kaggle_Merge_Demo.ipynb` — sections D–J added after the existing Stage 0
  cells; preprocessing is now cached.

Unchanged: `train.py`, `test.py`, `utils.py`, everything in `lib/`.

---

## 9. Running it

Local, in order:

```powershell
# Stage 0 — baseline (unchanged)
python train.py --data_path data/ --dims 12 --spec True --epochs 30 --batch_size 32 --save_path ckpt/
python test.py  --data_path data/ --dims 12 --spec True --mask_loss True --load_model 1 --load_path ckpt/TSRNet-latest.pt

# Stage 1 — forward diffusion, nothing trained
python experiments/exp01_forward_noise.py --split train --index 0 --tag normal
python experiments/exp01_forward_noise.py --split test --index 1 --normalize 1 --tag abnormal

# Stage 2 — smoke test, THEN train
python experiments/exp02_noise_prediction.py --batch_size 8 --steps 3
python training/train_diffusion.py --epochs 10 --batch_size 32 --T 100 --seed 668

# Stage 3 + 4 — score and timestep sweep
python experiments/exp03_step_analysis.py --repeats 3 --seed 668

# Stage 5 — fusion
python experiments/exp04_tsr_vs_noise.py --spec 1 --mask_loss 1 --repeats 3 --seed 668
python experiments/exp05_fusion.py --alpha 0.5
```

`--spec 1` must match how the TSR checkpoint was trained (`train.py --spec True`).
Passing the wrong one raises a clear error instead of producing meaningless scores.

On Kaggle, run `TSRNet_Kaggle_Merge_Demo.ipynb` section by section — **not** Run All on
the first pass. Sections A–C must complete and the Stage 0 AUC must be recorded before
Section D. The notebook checks CUDA and refuses to fall back to CPU, verifies the
`.npy` files and the diffusion modules before Section C, and caches both the
preprocessing and the diffusion checkpoint.

### Expected shapes

| tensor | shape |
|---|---|
| stored array | `[N, 5000, 12]` |
| dataloader window | `[B, 4800, 12]` |
| `x0`, `x_t`, `epsilon`, `epsilon_hat` | `[B, 12, 4800]` |
| encoder latent `z_t` | `[B, 50, 136]` |
| time embedding | `[B, 50]` |
| `A_t` | `[B]` |

---

## 10. Reproducibility

Every experiment records: the data split (PTB-XL fold 10 as test, `NORM`-only train),
the timestep grid, the checkpoint path, the scoring method, ROC-AUC and PR-AUC, and the
random seed. `results/scores_meta.json` carries the provenance of `scores.csv`, and
`ckpt/diffusion/train_history.json` carries the training curve and the full argument
list.

The ε draws are seeded per timestep (`seed + 1000 * t`), so a rerun with the same seed
reproduces the same scores.

---

## 11. Known limitations

**1. The train/test amplitude gap.** `preprocess.denoise_train` applies per-lead
min-max normalisation to [-1, 1]; `preprocess.denoise_test` **does not**. Measured on
the current arrays: train std ≈ 0.56 in [-1, 1], test std ≈ 0.20 with outliers beyond
±5. This is inherited from the original repository.

It matters much more for diffusion than for TSR-Net. The forward process mixes `x₀`
with unit-variance noise, so the effective signal-to-noise ratio at a given `t` depends
directly on the amplitude of `x₀`. A model trained at one amplitude and scored at
another is not a valid comparison.

The fix is applied **at evaluation time, not on disk**: `DiffusionECGSet(normalize=True)`
re-applies the same per-lead min-max to test windows, and it is on by default
(`--normalize_eval 1`). It is unsupervised and per-record, so no label information
leaks. The preprocessing outputs and the TSR-Net path are untouched, exactly as the
brief required. Pass `--normalize_eval 0` to see the uncorrected behaviour.

Caveat: min-max normalisation discards absolute amplitude, which is itself
diagnostically meaningful (e.g. low voltage, hypertrophy). Normalising per record may
therefore remove some signal. The clean long-term fix is to normalise the test set in
`preprocess.py` — which would change the Stage 0 baseline, so it is out of scope here
and flagged rather than done.

**2. Anomaly classes.** Only PTB-XL diagnostic superclasses are available. AF and PVC
labels do not exist in this pipeline and are not invented.

**3. `t` is fixed at scoring time.** Versions 4–5 of the step selector (sample-specific
and learnable) are not implemented. `A_N = Σ_t w_t A_t` weighted aggregation is not
implemented.

**4. The α sweep is tuned on the test set.** There is no validation split carrying
abnormal records, so the best α in the sweep is an upper bound. The fixed α = 0.5
result is the honest headline.

**5. Single seed.** Results are reported from one seed. Variance across seeds has not
been characterised.
