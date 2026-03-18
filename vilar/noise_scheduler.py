import torch
import math
import numpy as np
import matplotlib.pyplot as plt


class NoiseScheduler:
    
    def __init__(self, 
                 num_timesteps: int = 1000,
                 beta_schedule: str = "linear",
                 max_beta: float = 0.999,
                 eps: float = 1e-12,
                 device=None,
                 dtype=torch.float32):
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.max_beta = max_beta
        self.eps = eps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        self.beta = self._get_beta_schedule()
        self._compute_diffusion_constants()
    
    def _betas_for_alpha_bar(self, alpha_bar_fn):
        betas = []
        for i in range(self.num_timesteps):
            t1 = i / self.num_timesteps
            t2 = (i + 1) / self.num_timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), self.max_beta))
        return torch.tensor(betas, device=self.device, dtype=self.dtype)
    
    def _cosine_beta_schedule(self, s: float = 0.008):
        steps = self.num_timesteps
        t = torch.linspace(0, 1, steps + 1, device=self.device, dtype=self.dtype)
        alpha_bar = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = torch.clamp(alpha_bar, min=self.eps)
        
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        
        scale_factor = 1000 / self.num_timesteps
        betas = betas * scale_factor * 0.008
        
        return torch.clamp(betas, min=self.eps, max=self.max_beta)
    
    def _get_beta_schedule(self):
        if self.beta_schedule == "linear":
            scale = 1000 / self.num_timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            betas = torch.linspace(beta_start, beta_end, self.num_timesteps, device=self.device, dtype=self.dtype)
            
        elif self.beta_schedule == "quadratic":
            scale = 1000 / self.num_timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            u = torch.linspace(0.0, 1.0, self.num_timesteps, device=self.device, dtype=self.dtype)
            betas = beta_start + (beta_end - beta_start) * (u ** 2)
            
        elif self.beta_schedule == "cosine":
            betas = self._cosine_beta_schedule()
            
        else:
            raise ValueError(f"Invalid beta_schedule '{self.beta_schedule}'. Choose 'linear', 'quadratic', or 'cosine'.")
        betas = torch.clamp(betas, min=self.eps, max=self.max_beta)
        return betas
    
    def _compute_diffusion_constants(self):
        with torch.no_grad():
            self.alpha = 1.0 - self.beta
            self.alpha_hat = torch.cumprod(torch.clamp(self.alpha, min=self.eps), dim=0)
            
            self.sqrt_alpha = torch.sqrt(torch.clamp(self.alpha, min=self.eps))
            self.sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - self.alpha_hat, min=self.eps))
            
            alpha_hat_tm1 = torch.cat([
                torch.ones(1, device=self.device, dtype=self.dtype),
                self.alpha_hat[:-1]
            ], dim=0)
            posterior_var = self.beta * (1.0 - alpha_hat_tm1) / torch.clamp(1.0 - self.alpha_hat, min=self.eps)
            self.posterior_var = torch.clamp(posterior_var, min=self.eps)
            self.posterior_std = torch.sqrt(self.posterior_var)

    def register_to_model(self, model):
        if hasattr(model, 'set_noise_schedule'):
            model.set_noise_schedule(self.beta, eps=self.eps)
        else:
            model.register_buffer('beta', self.beta)
            model.register_buffer('alpha', self.alpha)
            model.register_buffer('alpha_hat', self.alpha_hat)
            model.register_buffer('sqrt_alpha', self.sqrt_alpha)
            model.register_buffer('sqrt_one_minus_alpha_bar', self.sqrt_one_minus_alpha_bar)
            model.register_buffer('posterior_std', self.posterior_std)
            
            model.num_timesteps = self.num_timesteps
        
        return model
