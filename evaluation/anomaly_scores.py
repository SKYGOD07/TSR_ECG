import torch

def compute_noise_anomaly_score(epsilon, epsilon_hat):
    """
    epsilon: [B, C, L] true noise
    epsilon_hat: [B, C, L] predicted noise
    Returns: [B] anomaly score per sample (MSE over channel and temporal dimensions)
    """
    # A_t = mean((epsilon - epsilon_hat)^2) over channel and temporal dimensions
    mse = (epsilon - epsilon_hat).pow(2)
    score = mse.mean(dim=(1, 2))
    return score
