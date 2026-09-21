import torch
import torch.nn as nn

def extract(a, t, x_shape):
    """
    Extract some coefficients at specified timesteps,
    then reshape to [batch_size, 1, 1, ...] for broadcasting purposes.
    """
    b = t.shape[0]
    out = a.gather(-1, t.cpu())
    out = out.to(t.device)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class DiffusionSchedule(nn.Module):
    def __init__(self, T=100, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.T = T
        
        # We precompute beta, alpha, and alpha_bar
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Precompute sqrt(alpha_bar_t) and sqrt(1 - alpha_bar_t)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
    def add_noise(self, x_0, t, noise=None):
        """
        x_0: [B, C, L] (e.g. [B, 12, 5000])
        t: [B] tensor of timesteps (zero-indexed: 0 to T-1)
        noise: optional pre-sampled noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise
