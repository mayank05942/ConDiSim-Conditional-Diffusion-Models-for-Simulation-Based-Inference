import torch
import numpy as np
import math
import sbibm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
import random

def load_data(task_name, num_samples):
    task = sbibm.get_task(task_name)
    prior = task.get_prior()
    simulator = task.get_simulator()
    x_o = task.get_observation(num_observation=1)

    thetas = prior(num_samples=num_samples)
    y_data = simulator(thetas)

    theta_scaler = StandardScaler()
    y_scaler = StandardScaler()
    thetas = theta_scaler.fit_transform(thetas)
    y_data = y_scaler.fit_transform(y_data)

    thetas = torch.tensor(thetas, dtype=torch.float32)
    y_data = torch.tensor(y_data, dtype=torch.float32)

    return thetas, y_data, theta_scaler, y_scaler, task

def create_dataloaders(thetas,
                       y_data,
                       batch_size):
    dataset = TensorDataset(thetas, y_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def cosine_beta_schedule(num_steps: int,
                         s: float = 0.008,
                         max_beta: float = 0.999,
                         device=None,
                         dtype=torch.float32):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    eps = 1e-12
    
    betas = []
    for i in range(num_steps):
        t1 = i / num_steps
        t2 = (i + 1) / num_steps
        alpha_bar_t1 = math.cos(((t1 + s) / (1 + s)) * (math.pi / 2)) ** 2
        alpha_bar_t2 = math.cos(((t2 + s) / (1 + s)) * (math.pi / 2)) ** 2
        alpha_bar_t1 = max(alpha_bar_t1, eps)
        beta = min(1 - (alpha_bar_t2 / alpha_bar_t1), max_beta)
        betas.append(beta)
    
    betas = torch.tensor(betas, device=device, dtype=dtype)
    return torch.clamp(betas, min=eps, max=max_beta)

def noise_schedule(num_timesteps,
                  beta_start=1e-4,
                  beta_end=0.02,
                  beta_schedule="linear",
                  device=None,
                  dtype=torch.float32,
                  max_beta: float = 0.999):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    if beta_schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device, dtype=dtype)
    elif beta_schedule == "quadratic":
        u = torch.linspace(0.0, 1.0, num_timesteps, device=device, dtype=dtype)
        betas = beta_start + (beta_end- beta_start) * (u ** 2)
    elif beta_schedule == "cosine":
        betas = cosine_beta_schedule(num_timesteps, device=device, dtype=dtype, max_beta=max_beta)
        
    else:
        raise ValueError("Invalid beta_schedule. Choose 'linear', 'quadratic', or 'cosine'.")
    
    return betas

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

