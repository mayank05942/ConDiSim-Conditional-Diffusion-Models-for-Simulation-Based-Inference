import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimeEmbeddingModule(nn.Module):
    def __init__(self, code_dim: int = 64, sin_dim: int = 64, hidden_dim: int = 256):
        self.sin_dim = sin_dim
        self.code_dim = code_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.sin_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, code_dim),
        )
        self.initialize()

    def initialize(self):
        pass

    def forward(self, t):
        t_emb = timestep_embedding(t, self.sin_dim)
        return self.mlp(t_emb)


class TimeSeriesEncoder(nn.Module):
    def __init__(self, in_channels=3, seq_length=200, code_dim=128, hidden_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.code_dim = code_dim
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),
        )
        
        self.flattened_size = 256 * 12
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, code_dim),
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc_layers(x)
        
        return x


class TimedependentFILM(nn.Module):
    def __init__(self, cond_dim: int = 128, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.pre_norm = nn.LayerNorm(cond_dim)

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, 4*out_dim),
            nn.SiLU(),
            nn.Linear(4*out_dim, out_dim * 2),
        )

        with torch.no_grad():
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def initialize(self):
        pass

    def forward(self, c: torch.Tensor):
        gb = self.mlp(self.pre_norm(c))
        return gb.chunk(2, dim=-1)


class ReverseDiffusionBlock(nn.Module):
    def __init__(self, hidden_dim=128, mlp_dim=512):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        h = 4 * hidden_dim
        self.post_film_mlp = nn.Sequential(
            nn.Linear(hidden_dim, h),
            nn.SiLU(),
            nn.Linear(h, hidden_dim),
        )
        with torch.no_grad():
            nn.init.zeros_(self.post_film_mlp[-1].weight)
            nn.init.zeros_(self.post_film_mlp[-1].bias)

        self.initialize()

    def initialize(self):
        pass

    def apply_film(self, x, gamma, beta):
        return x * (1 + gamma) + beta

    def forward(self, x, gamma, beta):
        residual = x
        h = self.apply_film(self.norm(x), gamma, beta)
        h = self.post_film_mlp(h)
        return h + residual


class DiffusionModel(nn.Module):
    def __init__(self, theta_dim, y_channels, y_seq_length, cfg_dropout_prob=0.2):
        super().__init__()
        self.theta_dim = theta_dim
        self.y_channels = y_channels
        self.y_seq_length = y_seq_length
        self.cfg_dropout_prob = cfg_dropout_prob
        
        self.hidden_dim = 128
        self.code_dim = 128
        self.num_blocks = 6
        
        self.num_timesteps = 0
        
        self.timembedding = TimeEmbeddingModule(
            code_dim=self.code_dim, 
            sin_dim=self.code_dim, 
            hidden_dim=128
        )
        
        self.y_enc = TimeSeriesEncoder(
            in_channels=y_channels,
            seq_length=y_seq_length,
            code_dim=self.code_dim,
            hidden_dim=128
        )
        
        self.null_y_embedding = nn.Parameter(torch.zeros(1, self.code_dim))
        
        self.theta_head = nn.Sequential(
            nn.Linear(theta_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
        )

        self.film_per_block = nn.ModuleList([
            TimedependentFILM(
                cond_dim=self.code_dim * 2,
                hidden_dim=self.hidden_dim,
                out_dim=self.hidden_dim,
            )
            for _ in range(self.num_blocks)
        ])

        self.theta_conditioned_blocks = nn.ModuleList([
            ReverseDiffusionBlock(
                hidden_dim=self.hidden_dim,
                mlp_dim=128,
            )
            for _ in range(self.num_blocks)
        ])

        self.theta_tail = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, theta_dim),
        )
        with torch.no_grad():
            last_linear = self.theta_tail[-1]
            if isinstance(last_linear, nn.Linear):
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)
        
        self.initialize()

    def initialize(self):
        pass

    def forward(self, theta_t, y, t, force_drop_condition=False):
        batch_size = theta_t.shape[0]
        device = theta_t.device
        
        t_code = self.timembedding(t)
        
        if force_drop_condition:
            y_code = self.null_y_embedding.expand(batch_size, -1)
        else:
            y_code_real = self.y_enc(y)
            if self.training and self.cfg_dropout_prob > 0.0:
                drop = (torch.rand(batch_size, 1, device=device) < self.cfg_dropout_prob)
                y_code_null = self.null_y_embedding.expand(batch_size, -1)
                y_code = torch.where(drop, y_code_null, y_code_real)
            else:
                y_code = y_code_real
        
        c = torch.cat([y_code, t_code], dim=-1)
        
        h = self.theta_head(theta_t)
        
        for i, blk in enumerate(self.theta_conditioned_blocks):
            gamma, beta = self.film_per_block[i](c)
            h = blk(h, gamma, beta)

        return self.theta_tail(h)
    
    def set_noise_schedule(self, beta: torch.Tensor, eps: float = 1e-12):
        with torch.no_grad():
            beta = beta.detach()
            self.register_buffer('beta', beta)
            alpha = 1.0 - beta
            self.register_buffer('alpha', alpha)
            alpha_hat = torch.cumprod(alpha, dim=0)
            self.register_buffer('alpha_hat', alpha_hat)
            self.num_timesteps = beta.numel()

            sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=eps))
            sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_hat, min=eps))
            self.register_buffer('sqrt_alpha', sqrt_alpha)
            self.register_buffer('sqrt_one_minus_alpha_bar', sqrt_one_minus_alpha_bar)

            alpha_hat_tm1 = torch.cat([
                torch.ones(1, device=alpha_hat.device, dtype=alpha_hat.dtype),
                alpha_hat[:-1]
            ], dim=0)
            posterior_var = beta * (1.0 - alpha_hat_tm1) / torch.clamp(1.0 - alpha_hat, min=eps)
            posterior_var = torch.clamp(posterior_var, min=eps)
            posterior_std = torch.sqrt(posterior_var)
            self.register_buffer('posterior_std', posterior_std)
