<h1 align="center">TSRNet: Simple Framework for Real-time ECG Anomaly Detection with Multimodal Time and Spectrogram Restoration Network (ISBI 2024)</h1>
<p align="center">
  <p align="center">
    <a href="https://tanbuinhat.github.io/"><strong>Nhat-Tan Bui</strong></a>
    ·
    <a href="https://dblp.org/pid/253/9950.html"><strong>Dinh-Hieu Hoang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=zsGhPHcAAAAJ&hl=vi&authuser=1"><strong>Thinh Phan</strong></a>
    ·
    <a href="https://www.fit.hcmus.edu.vn/~tmtriet/"><strong>Minh-Triet Tran</strong></a>
    .
    <a href="https://directory.hsc.wvu.edu/Profile/60996"><strong>Brijesh Patel</strong></a>
    .
    <a href="https://community.wvu.edu/~daadjeroh/"><strong>Donald Adjeroh</strong></a>
    .
    <a href="https://www.nganle.net/"><strong>Ngan Le</strong></a>
  </p>

  <h4 align="center"><a href="https://arxiv.org/abs/2312.10187">arXiv Paper</a> | <a href="docs/">Beginner-Friendly Project Guide</a> | <a href="docs/07_diffusion_noise_branch.md">Diffusion / Noise Branch</a> | <a href="docs/LOCAL_RUN_REPORT.md">Local Execution Report</a> | <a href="docs/output_result.md">Benchmark Report</a></h4>
  <div align="center"></div>

</p>

## Introduction
<image src="Images/main_architecture.png">
  
The electrocardiogram (ECG) is a valuable signal used to assess various aspects of heart health, such as heart rate and rhythm. It plays a crucial role in identifying cardiac conditions and detecting anomalies in ECG data. However, distinguishing between normal and abnormal ECG signals can be a challenging task. In this paper, we propose an approach that leverages anomaly detection to identify unhealthy conditions using solely normal ECG data for training. Furthermore, to enhance the information available and build a robust system, we suggest considering both the time series and time-frequency domain aspects of the ECG signal. As a result, we introduce a specialized network called the Multimodal Time and Spectrogram Restoration Network (TSRNet) designed specifically for detecting anomalies in ECG signals. TSRNet falls into the category of restoration-based anomaly detection and draws inspiration from both the time series and spectrogram domains. By extracting representations from both domains, TSRNet effectively captures the comprehensive characteristics of the ECG signal. This approach enables the network to learn robust representations with superior discrimination abilities, allowing it to distinguish between normal and abnormal ECG patterns more effectively. Furthermore, we introduce a novel inference method, termed Peak-based Error, that specifically focuses on ECG peaks, a critical component in detecting abnormalities. The experimental result on the large-scale dataset PTB-XL has demonstrated the effectiveness of our approach in ECG anomaly detection, while also prioritizing efficiency by minimizing the number of trainable parameters.

<table border="0">
  <tr>
    <td><image src="Images/cross_attention.png"></td>
    <td><image src="Images/peak_based.png"></td>
  </tr>
</table>

---

## 🚀 Local & Cloud Execution Guide

For a complete, beginner-friendly technical explanation, diagrams, and terminology breakdown of our local pipeline (including how we fixed the epoch loop range error), please see our **new** [Beginner Documentation Folder](docs/).

For the concise summary of the local metrics, please see [docs/LOCAL_RUN_REPORT.md](docs/LOCAL_RUN_REPORT.md).

### 1. Data Preprocessing (`preprocess.py`)
To process raw PhysioNet PTB-XL dataset WFDB files (`.dat` / `.hea`) directly:
```powershell
python preprocess.py --raw_path "path/to/PTBXL_folder" --out_path "data"
```
This generates `data/train.npy` (normal-only), `data/test.npy` (2,155 samples),
`data/label.npy` (binary 0/1) and `data/test_class.npy` (the PTB-XL diagnostic
superclass per test record, added for the diffusion branch's timestep analysis - the
first three files are unchanged).

### 2. Local Training (`train.py`)
To train the model locally:
```powershell
python train.py --data_path data/ --dims 12 --spec True --epochs 1 --batch_size 16 --save_path ckpt/
```

### 3. Local Evaluation (`test.py`)
To evaluate a compiled checkpoint model on the test dataset:
```powershell
python test.py --data_path data/ --dims 12 --spec True --load_model 1 --load_path ckpt/TSRNet-latest.pt
```

### 4. Diffusion / Noise Branch (research extension)

A DDPM diffusion branch has been added **alongside** the TSR-Net baseline. It does not
replace it: `train.py`, `test.py`, `preprocess.py` and everything in `lib/` behave
exactly as before.

Where TSR-Net masks part of the ECG and measures how badly it restores it, the
diffusion branch corrupts the ECG with a *known* amount of Gaussian noise and asks the
network to name the noise it added:

```
x_t     = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon,   epsilon ~ N(0, I)
eps_hat = g( TSR_encoder(x_t), TimeEmbedding(t) )
A_noise = mean( (epsilon - eps_hat)^2 )
A_final = alpha * A_TSR + (1 - alpha) * A_noise
```

The predictor **reuses TSR-Net's `Encoder1D`** rather than introducing a second
architecture (a separate instance with its own weights - the TSR checkpoints are
untouched), and is trained on **normal ECGs only**, so an abnormal record produces a
larger prior/posterior noise mismatch.

> **The HeartPy filtering in `preprocess.py` is not the diffusion noise.** That is
> preprocessing - *removing* baseline wander and mains hum. Diffusion noise is
> *injected* afterwards, deliberately, as the training target. Diffusion begins after
> the preprocessed arrays exist and never modifies them.

Full write-up, including the maths, the exact tensor shapes and the known limitations:
**[docs/07_diffusion_noise_branch.md](docs/07_diffusion_noise_branch.md)**.

Run the stages **in order** - each one is a gate for the next:

```powershell
# Stage 1 - forward diffusion only, nothing is trained
python experiments/exp01_forward_noise.py --split train --index 0 --tag normal
python experiments/exp01_forward_noise.py --split test --index 1 --normalize 1 --tag abnormal

# Stage 2 - smoke test FIRST, then train on normal ECGs only
python experiments/exp02_noise_prediction.py --batch_size 8 --steps 3
python training/train_diffusion.py --epochs 10 --batch_size 32 --T 100 --seed 668

# Stage 3 + 4 - noise anomaly score, and which timesteps are informative
python experiments/exp03_step_analysis.py --repeats 3 --seed 668

# Stage 5 - fuse the TSR score with the noise score
python experiments/exp04_tsr_vs_noise.py --spec 1 --mask_loss 1 --repeats 3 --seed 668
python experiments/exp05_fusion.py --alpha 0.5
```

`--spec 1` must match how the TSR checkpoint was trained (`train.py --spec True`);
a mismatch raises a clear error rather than producing meaningless scores.

Diffusion checkpoints go to `ckpt/diffusion/`, kept separate from the TSR-Net
checkpoints in `ckpt/`. Result tables go to `results/`, figures to `Images/`.

#### Ablation

| | TSR | Diffusion | Noise score | Adaptive steps | Status |
|---|---|---|---|---|---|
| **M1** | yes | - | - | - | implemented |
| **M2** | - | yes | yes | - | implemented |
| **M4** | yes | yes | yes | - | implemented |
| **M5** | yes | yes | yes | global `t*` | implemented |
| **M6** | yes | yes | yes | anomaly-specific | not implemented |
| **M7** | yes | yes | yes | learnable | not implemented |

M6/M7 are deliberately left for later. Building a learned timestep selector before the
fixed-`t` numbers are in hand would make any gain impossible to attribute.

#### Scientific ground rules baked into the code

- No AUC is hard-coded, and no improvement is claimed that was not measured - if
  fusion fails to beat the better single branch, the script prints exactly that.
- No timestep is called optimal until it has been evaluated; `best_timestep()` reports
  the argmax over the grid that actually ran, and labels it as such.
- The alpha sweep is tuned on the test set, so it is reported as an upper bound; the
  fixed alpha = 0.5 result is the headline.
- PTB-XL superclasses (`NORM/MI/STTC/CD/HYP`) come from `data/test_class.npy`, saved
  additively by `preprocess.py`. **AF and PVC labels do not exist in this pipeline and
  are never fabricated.**

#### Known limitation: train/test amplitude gap

`preprocess.denoise_train` normalises each lead to [-1, 1]; `preprocess.denoise_test`
does not (train std ~0.56, test std ~0.20). Diffusion SNR depends directly on the
amplitude of `x_0`, so this is corrected **at evaluation time, not on disk** -
`--normalize_eval 1` (the default) re-applies the same per-lead min-max to test
windows. The preprocessing outputs and the TSR-Net path are untouched. See
[docs/07_diffusion_noise_branch.md](docs/07_diffusion_noise_branch.md) section 11.

---

## ⚡ Cloud / Kaggle Notebooks

- **[TSRNet_Kaggle_Merge_Demo.ipynb](TSRNet_Kaggle_Merge_Demo.ipynb):** the main notebook. Auto-merges split PTB-XL uploads, then runs the full pipeline in labelled sections:

  | Section | Stage | What it does |
  |---|---|---|
  | A | - | Internet, GPU check (refuses to fall back to CPU), dependencies, repo |
  | B | - | Merge PTB-XL, preprocess (cached), verify the `.npy` files **and** the diffusion modules |
  | C | **0** | Existing TSR-Net baseline - train, test, record the baseline AUC |
  | D | **1** | DDPM forward diffusion (nothing trained) |
  | E | **1 test** | Noise visualisation at t = 1, 10, 25, 50, 75, 100 - normal vs abnormal |
  | F | **2** | Noise predictor + smoke test (the gate before training) |
  | G | **2 train** | Normal-ECG-only training (cached) + loss curve |
  | H | **3** | Noise anomaly score |
  | I | **4** | Timestep analysis |
  | J | **5** | TSR + noise fusion, M1/M2/M4 ablation |

  Run it **section by section** on the first pass, not Run All. Sections A-C must
  complete and the Stage 0 AUC must be recorded before Section D.

- **[TSRNet_Kaggle_Drive_Download.ipynb](TSRNet_Kaggle_Drive_Download.ipynb):** Notebook equipped with high-speed 16-connection parallel `aria2c` PhysioNet dataset downloading for automated 50-epoch GPU training on Kaggle. (Stage 0 baseline only.)
- **[TSRNet_Kaggle_Demo.ipynb](TSRNet_Kaggle_Demo.ipynb):** Notebook pre-configured for Kaggle Input Datasets. (Stage 0 baseline only.)

---

## Prerequisites
<ul>
  <li>Pytorch</li>
  <li>Torchvision</li>
  <li>Numpy</li>
  <li>SciPy</li>
  <li>wfdb</li>
  <li>heartpy</li>
  <li>scikit-learn</li>
  <li>pandas, matplotlib, tqdm (diffusion experiments and result tables)</li>
</ul>

## Datasets
To validate the effectiveness of our model, we conduct the experiments benchmark in PTB-XL dataset. We follow the same dataset preprocessing as in <a href="https://github.com/MediaBrain-SJTU/ECGAD">Jiang et al.</a>

## Citation
```bibtex
@article{tsrnet,
      title={TSRNet: Simple Framework for Real-time ECG Anomaly Detection with Multimodal Time and Spectrogram Restoration Network}, 
      author={Nhat-Tan Bui and Dinh-Hieu Hoang and Thinh Phan and Minh-Triet Tran and Brijesh Patel and Donald Adjeroh and Ngan Le},
      journal={arXiv:2312.10187},
      year={2023}
}
```

## Acknowledgment
A part of this code is adapted from these previous works: [Jiang et al.](https://github.com/MediaBrain-SJTU/ECGAD) and [Phan et al.](https://github.com/UARK-AICV/ECG_SSL_12Lead)
