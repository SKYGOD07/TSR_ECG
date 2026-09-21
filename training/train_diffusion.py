import os
import sys
import argparse
import random
import time
import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path so we can import TSRNet modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader import TrainSet
from diffusion.schedule import DiffusionSchedule
from models.diffusion_model import DiffusionNoisePredictor

def main(args):
    # Set seeds for reproducibility
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    use_cuda = torch.cuda.is_available()
    device = "cuda:" + args.gpu if use_cuda else 'cpu'
    print(f"Using device: {device}")
    
    # Load normal-only training data
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    dset = TrainSet(folder=args.data_path, fs=500, nperseg=125)
    train_loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    # Initialize Models
    schedule = DiffusionSchedule(T=args.T).to(device)
    model = DiffusionNoisePredictor(channels=args.dims).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {round(n_parameters * 1e-6, 2)} M")
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        # We only use time_ecg from the dataloader for this experiment
        # dataloader returns: time_instance, spectrogram_instance
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, (time_ecg, _) in progress_bar:
            optimizer.zero_grad()
            
            # Format: [B, 4800, 12] -> [B, 12, 4800]
            x0 = time_ecg.float().to(device).transpose(1, 2)
            b = x0.shape[0]
            
            # Randomly sample timestep t
            t = torch.randint(0, args.T, (b,), device=device).long()
            
            # Sample random Gaussian noise
            epsilon = torch.randn_like(x0)
            
            # Forward diffusion
            x_t, noise = schedule.add_noise(x0, t, noise=epsilon)
            
            # Predict noise
            epsilon_hat = model(x_t, t)
            
            # Compute MSE loss
            loss = nn.functional.mse_loss(epsilon_hat, epsilon)
            
            # Backprop
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * b
            progress_bar.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Train Epoch: {epoch+1} Average Loss: {avg_loss:.6f}')
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(args.save_path, f'DiffusionNet-{epoch}.pt'))
        
        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(args.save_path, 'DiffusionNet-latest.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diffusion Noise Predictor Training')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=50, help='maximum training epochs')
    parser.add_argument('--dims', type=int, default=12, help='dimension of the input data')
    parser.add_argument('--save_path', type=str, default='ckpt/diffusion/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimizer')
    parser.add_argument('--seed', type=int, default=668, help='manual seed')
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--T", type=int, default=100, help='Total diffusion timesteps')
    
    args = parser.parse_args()
    main(args)
