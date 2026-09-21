import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.schedule import DiffusionSchedule
from models.diffusion_model import DiffusionNoisePredictor

def main():
    print("Running Noise Prediction Smoke Test (Stage 2)")
    
    # Load dataset
    data_path = 'data'
    train_npy_path = os.path.join(data_path, 'train.npy')
    
    if not os.path.exists(train_npy_path):
        print(f"Error: Could not find {train_npy_path}. Make sure to run preprocessing first.")
        return
        
    print(f"Loading {train_npy_path}...")
    train_data = np.load(train_npy_path)
    
    # Take a batch of 8 ECGs
    batch_size = 8
    time_instance = train_data[:batch_size]
    time_instance = time_instance[:, 100:4900, :]  # Shape: (8, 4800, 12)
    
    # Convert to tensor and transpose to channel first
    # [B, 4800, 12] -> [B, 12, 4800]
    x0 = torch.tensor(time_instance, dtype=torch.float32).transpose(1, 2)
    
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    x0 = x0.to(device)
    
    print(f"Device: {device}")
    
    schedule = DiffusionSchedule(T=100).to(device)
    model = DiffusionNoisePredictor(channels=12).to(device)
    
    # Sample random timesteps
    t = torch.randint(0, 100, (batch_size,), device=device).long()
    
    # Generate Gaussian noise
    epsilon = torch.randn_like(x0)
    
    # Forward diffusion
    x_t, _ = schedule.add_noise(x0, t, noise=epsilon)
    
    # Predict noise
    epsilon_hat = model(x_t, t)
    
    # Compute MSE loss
    loss = nn.functional.mse_loss(epsilon_hat, epsilon)
    
    # Print shapes
    print(f"x0 shape:          {x0.shape}")
    print(f"xt shape:          {x_t.shape}")
    print(f"epsilon shape:     {epsilon.shape}")
    print(f"epsilon_hat shape: {epsilon_hat.shape}")
    print(f"loss:              {loss.item()}")
    
    # Assertions
    assert x_t.shape == x0.shape, "xt shape mismatch"
    assert epsilon.shape == x0.shape, "epsilon shape mismatch"
    assert epsilon_hat.shape == epsilon.shape, "epsilon_hat shape mismatch"
    
    print("Stage 2 Smoke Test Passed!")

if __name__ == '__main__':
    main()
