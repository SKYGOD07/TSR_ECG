import os
import sys
import copy
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import TestSet
from lib.TSRNet_time import TSRNet_time
from diffusion.schedule import DiffusionSchedule
from models.diffusion_model import DiffusionNoisePredictor
from evaluation.anomaly_scores import compute_noise_anomaly_score

def min_max_normalize(scores):
    s_min = scores.min()
    s_max = scores.max()
    if s_max == s_min:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)

def main():
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. Dataset
    data_path = 'data'
    test_loader = torch.utils.data.DataLoader(
        TestSet(folder=data_path, fs=500, nperseg=125),
        batch_size=1, shuffle=False
    )
    labels = np.load(os.path.join(data_path, 'label.npy')).astype(int)
    
    # 2. Models
    tsr_model = TSRNet_time(enc_in=12).to(device)
    tsr_ckpt = 'ckpt/TSRNet-latest.pt'
    if os.path.exists(tsr_ckpt):
        tsr_model.load_state_dict(torch.load(tsr_ckpt, map_location=device)['model_state_dict'])
    tsr_model.eval()
    
    diff_model = DiffusionNoisePredictor(channels=12).to(device)
    diff_ckpt = 'ckpt/diffusion/DiffusionNet-latest.pt'
    if os.path.exists(diff_ckpt):
        diff_model.load_state_dict(torch.load(diff_ckpt, map_location=device)['model_state_dict'])
    diff_model.eval()
    schedule = DiffusionSchedule(T=100).to(device)
    
    tsr_scores = []
    noise_scores = []
    best_t = 50 
    
    print("Computing anomaly scores...")
    with torch.no_grad():
        for i, (time_ecg, spectrogram_ecg, r_index) in tqdm(enumerate(test_loader), total=len(labels)):
            time_length = time_ecg.shape[1]
            time_ecg = time_ecg.float().to(device)
            mask_time = copy.deepcopy(time_ecg)
            
            mask = torch.zeros((1, time_length, 1), dtype=torch.bool).to(device)
            patch_interval_time = 4800 // 30
            for j in range(100 // 30):
                for k in range(30):
                    cut_idx = 48*j + patch_interval_time*k
                    mask[:, cut_idx:cut_idx+48] = 1
            mask_time = torch.mul(mask_time, ~mask)
            
            gen_time, time_var = tsr_model(mask_time)
            time_err = (gen_time - time_ecg) ** 2
            l_time = torch.mean(torch.exp(-time_var)*time_err)
            
            tsr_score = l_time.detach().cpu().item()
            tsr_scores.append(tsr_score)
            
            x0 = time_ecg.transpose(1, 2)
            t = torch.tensor([best_t - 1], device=device, dtype=torch.long)
            epsilon = torch.randn_like(x0)
            
            x_t, _ = schedule.add_noise(x0, t, noise=epsilon)
            epsilon_hat = diff_model(x_t, t)
            
            n_score = compute_noise_anomaly_score(epsilon, epsilon_hat).item()
            noise_scores.append(n_score)
            
    tsr_scores = np.array(tsr_scores)
    noise_scores = np.array(noise_scores)
    
    tsr_norm = min_max_normalize(tsr_scores)
    noise_norm = min_max_normalize(noise_scores)
    
    auc_tsr = roc_auc_score(labels, tsr_norm)
    auc_noise = roc_auc_score(labels, noise_norm)
    
    print("\n--- Model Comparison ---")
    print(f"M1 (TSR-Net Only) AUC:       {auc_tsr:.4f}")
    print(f"M2 (Noise Only, t={best_t}) AUC: {auc_noise:.4f}")

if __name__ == '__main__':
    main()
