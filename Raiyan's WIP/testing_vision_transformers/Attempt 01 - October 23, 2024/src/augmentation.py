# augmentation.py (New File)
import torch

def add_noise(data, std_dev=0.01):
    """Add Gaussian noise to the radar input."""
    noise = torch.randn_like(data) * std_dev
    return data + noise
