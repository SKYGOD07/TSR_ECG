import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        """
        t: [B] tensor of timesteps
        Returns: [B, dim] tensor of embeddings
        """
        t = t.view(-1, 1).float()
        return self.mlp(t)
