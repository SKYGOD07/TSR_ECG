import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion.schedule import DiffusionSchedule
from models.diffusion_model import DiffusionNoisePredictor
from evaluation.anomaly_scores import compute_noise_anomaly_score
from dataloader import TestSet

def main():
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load test dataset
    data_path = 'data'
    test_loader = torch.utils.data.DataLoader(
        TestSet(folder=data_path, fs=500, nperseg=125),
        batch_size=32, shuffle=False
    )
    labels = np.load(os.path.join(data_path, 'label.npy'))
    
    # Load Models (Assume they are trained for this script)
    ckpt_path = 'ckpt/diffusion/DiffusionNet-latest.pt'
    schedule = DiffusionSchedule(T=100).to(device)
    model = DiffusionNoisePredictor(channels=12).to(device)
    
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Warning: Checkpoint {ckpt_path} not found. Running with untrained model.")
        
    model.eval()
    
    # Timesteps to evaluate (convert to 0-indexed)
    timesteps_to_test = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    results = []
    
    # Pre-compute labels
    test_labels = labels.astype(int)
    
    with torch.no_grad():
        for t_val in timesteps_to_test:
            print(f"Evaluating t={t_val}...")
            t_idx = t_val - 1
            all_scores = []
            
            for time_ecg, _, _ in tqdm(test_loader, leave=False):
                x0 = time_ecg.float().to(device).transpose(1, 2)
                b = x0.shape[0]
                
                t = torch.full((b,), t_idx, device=device, dtype=torch.long)
                epsilon = torch.randn_like(x0)
                
                x_t, _ = schedule.add_noise(x0, t, noise=epsilon)
                epsilon_hat = model(x_t, t)
                
                scores = compute_noise_anomaly_score(epsilon, epsilon_hat)
                all_scores.extend(scores.cpu().numpy())
                
            all_scores = np.array(all_scores)
            
            # Normal vs Abnormal
            normal_scores = all_scores[test_labels == 0]
            abnormal_scores = all_scores[test_labels == 1]
            
            mean_normal = normal_scores.mean()
            mean_abnormal = abnormal_scores.mean()
            diff = mean_abnormal - mean_normal
            
            # AUC
            try:
                auc = roc_auc_score(test_labels, all_scores)
            except ValueError:
                auc = 0.5
                
            results.append({
                't': t_val,
                'mean_normal': mean_normal,
                'mean_abnormal': mean_abnormal,
                'diff': diff,
                'auc': auc
            })
            
    df = pd.DataFrame(results)
    print("\n--- Timestep Analysis Results ---")
    print(df.to_string(index=False))
    
    # Plotting
    os.makedirs('Images', exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(df['t'], df['mean_normal'], label='Normal', marker='o')
    plt.plot(df['t'], df['mean_abnormal'], label='Abnormal', marker='o')
    plt.xlabel("Diffusion Timestep t")
    plt.ylabel("Mean Noise Anomaly Score (MSE)")
    plt.title("Noise Anomaly Score by Timestep")
    plt.legend()
    plt.grid(True)
    plt.savefig('Images/step_analysis_scores.png')
    
    plt.figure(figsize=(10, 5))
    plt.plot(df['t'], df['auc'], label='AUC', marker='x', color='green')
    plt.xlabel("Diffusion Timestep t")
    plt.ylabel("ROC-AUC")
    plt.title("AUC by Timestep")
    plt.grid(True)
    plt.savefig('Images/step_analysis_auc.png')
    
    print("Saved plots to Images/")

if __name__ == '__main__':
    main()
