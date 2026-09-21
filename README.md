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

  <h4 align="center"><a href="https://arxiv.org/abs/2312.10187">arXiv Paper</a> | <a href="docs/">Beginner-Friendly Project Guide</a> | <a href="docs/LOCAL_RUN_REPORT.md">Local Execution Report</a> | <a href="docs/output_result.md">Benchmark Report</a></h4>
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
This generates `data/train.npy` (14,241 samples), `data/test.npy` (2,155 samples), and `data/label.npy`.

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

### 4. Diffusion Noise Predictor (New Experimental Branch)
A new diffusion-based noise prediction branch has been added as a research experiment. This branch uses the existing TSRNet 1D CNN encoder to predict the added noise at various timesteps of a forward diffusion process.

To test the forward diffusion noise addition:
```powershell
python experiments/exp01_forward_noise.py
```

To run a smoke test of the noise predictor architecture:
```powershell
python experiments/exp02_noise_prediction.py
```

To train the noise predictor on normal ECGs:
```powershell
python training/train_diffusion.py --epochs 50 --batch_size 32
```

To evaluate noise anomaly scores across timesteps:
```powershell
python experiments/exp03_step_analysis.py
```

To establish baselines and fuse the original TSRNet anomaly scores with the new diffusion anomaly scores:
```powershell
python experiments/exp04_tsr_vs_noise.py
python experiments/exp05_fusion.py
```

---

## ⚡ Cloud / Kaggle Notebooks

- **[TSRNet_Kaggle_Drive_Download.ipynb](TSRNet_Kaggle_Drive_Download.ipynb):** Notebook equipped with high-speed 16-connection parallel `aria2c` PhysioNet dataset downloading for automated 50-epoch GPU training on Kaggle.
- **[TSRNet_Kaggle_Demo.ipynb](TSRNet_Kaggle_Demo.ipynb):** Notebook pre-configured for Kaggle Input Datasets.

---

## Prerequisites
<ul>
  <li>Pytorch</li>
  <li>Torchvision</li>
  <li>Numpy</li>
  <li>SciPy</li>
  <li>wfdb</li>
  <li>scikit-learn</li>
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
