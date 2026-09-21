import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path so we can import TSRNet modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.schedule import DiffusionSchedule

def main():
    print("Running Forward Diffusion Smoke Test (Stage 1)")
    
    # Load dataset
    data_path = 'data'
    train_npy_path = os.path.join(data_path, 'train.npy')
    
    if not os.path.exists(train_npy_path):
        print(f"Error: Could not find {train_npy_path}. Make sure to run preprocessing first.")
        return
        
    print(f"Loading {train_npy_path}...")
    train_data = np.load(train_npy_path)
    print(f"Loaded train_data with shape: {train_data.shape}")
    
    # Take the first ECG
    time_instance = train_data[0]
    time_instance = time_instance[100:4900, :]  # Shape: (4800, 12)
    
    # Convert to tensor and add batch dimension, transpose to channel first
    # [B, 5000, 12] -> [B, 4800, 12] -> [B, 12, 4800]
    x0 = torch.tensor(time_instance, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
    
    print(f"x0 shape: {x0.shape}")
    
    # Assertions
    assert len(x0.shape) == 3, "x0 must be 3D tensor [B, C, L]"
    assert x0.shape[1] == 12, "x0 must have 12 channels (leads)"
    assert not torch.isnan(x0).any(), "x0 contains NaNs"
    assert not torch.isinf(x0).any(), "x0 contains Infs"
    
    # Initialize Diffusion Schedule
    schedule = DiffusionSchedule(T=100)
    
    timesteps_to_test = [0, 9, 24, 49, 74, 99]  # 0-indexed corresponding to t=1, 10, 25, 50, 75, 100
    display_t = [1, 10, 25, 50, 75, 100]
    
    plt.figure(figsize=(15, 12))
    
    for i, (t_idx, disp_t) in enumerate(zip(timesteps_to_test, display_t)):
        t = torch.tensor([t_idx], dtype=torch.long)
        xt, noise = schedule.add_noise(x0, t)
        
        # Assertions
        assert xt.shape == x0.shape, f"xt shape {xt.shape} doesn't match x0 shape {x0.shape}"
        assert noise.shape == x0.shape, f"noise shape {noise.shape} doesn't match x0 shape {x0.shape}"
        assert not torch.isnan(xt).any(), f"xt contains NaNs at t={disp_t}"
        assert not torch.isinf(xt).any(), f"xt contains Infs at t={disp_t}"
        
        # Plot lead II (index 1)
        lead_idx = 1
        x_np = xt[0, lead_idx, :].numpy()
        
        plt.subplot(len(timesteps_to_test) + 1, 1, i + 1)
        if i == 0:
            # For t=1, also show clean signal
            plt.plot(x0[0, lead_idx, :].numpy(), label='Clean x0', color='blue', alpha=0.5)
        plt.plot(x_np, label=f'Noisy xt (t={disp_t})', color='red', alpha=0.8)
        plt.legend(loc='upper right')
        plt.ylabel("Amplitude")
        plt.title(f"Forward Diffusion at t={disp_t}")
        
    plt.xlabel("Time steps")
    plt.tight_layout()
    
    # Create Images dir if it doesn't exist
    os.makedirs('Images', exist_ok=True)
    plt.savefig('Images/forward_diffusion_smoke_test.png')
    print("Saved visualization to Images/forward_diffusion_smoke_test.png")
    print("Stage 1 Smoke Test Passed!")

if __name__ == '__main__':
    main()
