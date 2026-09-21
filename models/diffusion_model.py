import torch
import torch.nn as nn
from lib.modules import Encoder1D
from .time_embedding import TimeEmbedding

class NoiseDecoder1D(nn.Module):
    def __init__(self, nc):
        super(NoiseDecoder1D, self).__init__()
        ngf = 32
        self.main=nn.Sequential(
            nn.ConvTranspose1d(50, ngf*16, 15, 1, 0),
            nn.BatchNorm1d(ngf*16),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 16, ngf * 8, 4, 2, 1),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 4, ngf*2, 4, 2, 1),
            nn.BatchNorm1d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 2, ngf , 4, 2, 1),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf, nc, 4, 2, 1)
            # Removed Tanh here to predict unbounded Gaussian noise
        )

    def forward(self, input):
        output = self.main(input)
        return output

class DiffusionNoisePredictor(nn.Module):
    def __init__(self, channels=12, time_emb_dim=50):
        super().__init__()
        # 1. Feature extraction using existing TSR-Net encoder
        self.encoder = Encoder1D(channels)
        
        # 2. Timestep embedding (dim matches the encoder's output channel size, 50)
        self.time_embedding = TimeEmbedding(time_emb_dim)
        
        # 3. Noise prediction head (Decoder without Tanh)
        self.decoder = NoiseDecoder1D(channels)
        
    def forward(self, x_t, t):
        """
        x_t: [B, 12, L]
        t: [B]
        Returns: [B, 12, L] predicted noise epsilon_hat
        """
        # Encode noisy input
        z_t = self.encoder(x_t)  # [B, 50, 136]
        
        # Get time embedding
        t_emb = self.time_embedding(t)  # [B, 50]
        
        # Condition: add time embedding across sequence length dimension
        z_t = z_t + t_emb.unsqueeze(-1)  # [B, 50, 136] + [B, 50, 1]
        
        # Decode back to predict noise
        epsilon_hat = self.decoder(z_t)  # [B, 12, L]
        
        return epsilon_hat
