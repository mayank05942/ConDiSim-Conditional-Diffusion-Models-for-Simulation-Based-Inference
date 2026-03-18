import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Copy of the timestep_embedding function from core/train_utils.py
def timestep_embedding(timesteps, dim, max_period=10000):
    # DO NOT TOUCH
    """
    Create sinusoidal timestep embeddings for batch of timesteps. 
    
    :timesteps: a 1-D Tensor of N indices, one per batch element.
                      The timestep index for each element in the batch.
    :dim: Size of the embedding vector for each timestep.
    :max_period: Constant that sets the slowest frequency in the sinusoidal curves.  
    Larger values give the model a wider range of frequencies to represent long timesteps.
    
    :return: an [N x dim] Tensor of time embeddings.
    """
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
    """Time embedding module that converts diffusion timesteps to feature vectors.
    
    Uses the timestep_embedding function and projects it to the desired dimension.
    """
    def __init__(self, code_dim: int = 64, sin_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        # Sinusoidal embedding of size sin_dim, then learnable projection to code_dim
        self.sin_dim = sin_dim
        self.code_dim = code_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.sin_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, code_dim),
        )
        self.initialize()

    def initialize(self):
        """Initialize weights using shared initialization function."""
        # Note: For this small model, PyTorch defaults are usually sufficient.
        pass

    def forward(self, t):
        """Convert timesteps to embeddings.
        
        Args:
            t: Tensor of shape [batch_size] containing timestep indices
            
        Returns:
            Tensor of shape [batch_size, code_dim] containing time embeddings
        """
        t_emb = timestep_embedding(t, self.sin_dim)
        return self.mlp(t_emb)  # [batch_size, code_dim]


class TimeSeriesEncoder(nn.Module):
    """Encoder for time series data using Conv1D layers.
    
    This is the only component that differs from the core implementation,
    as it handles 3D time series data instead of flat vectors.
    """
    def __init__(self, in_channels=3, seq_length=200, code_dim=128, hidden_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.code_dim = code_dim
        
        # Conv1D layers for feature extraction
        self.conv_layers = nn.Sequential(
            # Layer 1: (batch_size, 3, 200) -> (batch_size, 32, 100)
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),
            
            # Layer 2: (batch_size, 32, 100) -> (batch_size, 64, 50)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),
            
            # Layer 3: (batch_size, 64, 50) -> (batch_size, 128, 25)
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),
            
            # Layer 4: (batch_size, 128, 25) -> (batch_size, 256, 12)
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),
        )
        
        # Calculate the flattened size after convolutions
        # After 4 max pooling layers with kernel_size=2, the sequence length is reduced by 2^4 = 16
        # So 200 / 16 = 12.5, rounded to 12
        self.flattened_size = 256 * 12
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, code_dim),
        )
        
    def forward(self, x):
        """
        Args:
            x: Time series data of shape (batch_size, in_channels, seq_length)
        Returns:
            Encoded features of shape (batch_size, code_dim)
        """
        # Apply Conv1D layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        return x


class TimedependentFILM(nn.Module):
    """Feature-wise Linear Modulation (FiLM).
    Input: c=[y_code|t_code] of shape [B, cond_dim]
    Output: gamma, beta each [B, out_dim]
    """
    def __init__(self, cond_dim: int = 128, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Light normalization to balance y_code vs t_code scales
        self.pre_norm = nn.LayerNorm(cond_dim)

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, 4*out_dim),
            nn.SiLU(),
            nn.Linear(4*out_dim, out_dim * 2),  # -> [B, 2*out_dim]
        )

        # Zero-init last layer so block starts near identity (gamma≈0, beta≈0)
        with torch.no_grad():
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def initialize(self):
        # Keep for compatibility; no-op by default.
        pass

    def forward(self, c: torch.Tensor):
        gb = self.mlp(self.pre_norm(c))
        return gb.chunk(2, dim=-1)  # gamma, beta


class ReverseDiffusionBlock(nn.Module):
    """Core building block for reverse diffusion model.
    
    Applies normalization, FiLM conditioning, and residual connections.
    """
    def __init__(self, hidden_dim=128, mlp_dim=512):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        h = 4 * hidden_dim
        
        # Block MLP expansion with dropout
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
        """Initialize weights using shared initialization function."""
        pass

    def apply_film(self, x, gamma, beta):
        """Apply FiLM conditioning.
        
        Args:
            x: Input tensor
            gamma: Scaling parameter
            beta: Shift parameter
            
        Returns:
            FiLM-conditioned tensor
        """
        return x * (1 + gamma) + beta

    def forward(self, x, gamma, beta):
        """Process input with FiLM conditioning.
        
        Args:
            x: Input tensor of shape [batch_size, hidden_dim]
            gamma: Scaling parameter of shape [batch_size, hidden_dim]
            beta: Shift parameter of shape [batch_size, hidden_dim]
            
        Returns:
            Processed tensor of shape [batch_size, hidden_dim]
        """
        # AdaLN-style block: normalize, FiLM, MLP, then residual add
        residual = x
        h = self.apply_film(self.norm(x), gamma, beta)
        h = self.post_film_mlp(h)
        return h + residual


class DiffusionModel(nn.Module):
    """Reverse diffusion model for posterior sampling.
    
    Estimates the theta noise given y and t (reverse diffusion process).
    Supports Classifier-Free Guidance for improved sample quality.
    
    This is the Vilar version of ReverseDiffusionModel, adapted for time series data.
    """
    def __init__(self, theta_dim, y_channels, y_seq_length, cfg_dropout_prob=0.2):
        """Initialize the reverse diffusion model.
        
        Args:
            theta_dim: Dimension of the parameter space
            y_channels: Number of channels in the time series
            y_seq_length: Length of the time series
            cfg_dropout_prob: Probability of dropping condition during training for CFG
        """
        super().__init__()
        self.theta_dim = theta_dim  # dimension of theta
        self.y_channels = y_channels  # number of channels in time series
        self.y_seq_length = y_seq_length  # length of time series
        self.cfg_dropout_prob = cfg_dropout_prob  # probability of dropping condition during training
        
        # Model hyperparameters - set manually as per requirements
        self.hidden_dim = 128
        self.code_dim = 128
        self.num_blocks = 6
        
        # Will be set by noise scheduler
        self.num_timesteps = 0
        
        # Time embedding
        self.timembedding = TimeEmbeddingModule(
            code_dim=self.code_dim, 
            sin_dim=self.code_dim, 
            hidden_dim=128
        )
        
        # Y encoder for time series data
        self.y_enc = TimeSeriesEncoder(
            in_channels=y_channels,
            seq_length=y_seq_length,
            code_dim=self.code_dim,
            hidden_dim=128
        )
        
        # Null embedding for unconditional generation (CFG)
        self.null_y_embedding = nn.Parameter(torch.zeros(1, self.code_dim))
        
        # Theta projection (use LayerNorm for stability regardless of input scaling)
        self.theta_head = nn.Sequential(
            nn.Linear(theta_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
        )

        # Per-block FiLM conditioning
        self.film_per_block = nn.ModuleList([
            TimedependentFILM(
                cond_dim=self.code_dim * 2,  # y_code + t_code
                hidden_dim=self.hidden_dim,
                out_dim=self.hidden_dim,
            )
            for _ in range(self.num_blocks)
        ])

        # Diffusion blocks
        self.theta_conditioned_blocks = nn.ModuleList([
            ReverseDiffusionBlock(
                hidden_dim=self.hidden_dim,
                mlp_dim=128,
            )
            for _ in range(self.num_blocks)
        ])

        # Output head (keep LayerNorm for stability; do not rely solely on input scaling)
        self.theta_tail = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, theta_dim),
        )
        # Zero-initialize the final output layer for stable start of training
        with torch.no_grad():
            last_linear = self.theta_tail[-1]
            if isinstance(last_linear, nn.Linear):
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)
        
        self.initialize()

    def initialize(self):
        """Initialize weights using shared initialization function."""
        pass

    def forward(self, theta_t, y, t, force_drop_condition=False):
        """Forward pass of the reverse diffusion model.
        
        Args:
            theta_t: Noisy parameters at timestep t, shape [batch_size, theta_dim]
            y: Time series data, shape [batch_size, y_channels, y_seq_length]
            t: Timestep indices, shape [batch_size]
            force_drop_condition: If True, always use unconditional path (for CFG)
            
        Returns:
            Predicted noise, shape [batch_size, theta_dim]
        """
        batch_size = theta_t.shape[0]
        device = theta_t.device
        
        # Encode time
        t_code = self.timembedding(t)  # [batch_size, code_dim]
        
        # --- Classifier-Free Guidance dropout during training ---
        if force_drop_condition:
            # unconditional path
            y_code = self.null_y_embedding.expand(batch_size, -1)  # [B, code_dim]
        else:
            y_code_real = self.y_enc(y)                              # [B, code_dim]
            if self.training and self.cfg_dropout_prob > 0.0:
                drop = (torch.rand(batch_size, 1, device=device) < self.cfg_dropout_prob)
                y_code_null = self.null_y_embedding.expand(batch_size, -1)
                y_code = torch.where(drop, y_code_null, y_code_real)
            else:
                y_code = y_code_real
        
        # Build condition vector c
        c = torch.cat([y_code, t_code], dim=-1)  # [batch_size, 2*code_dim]
        
        # Project theta to a high dim space
        h = self.theta_head(theta_t)  # [batch_size, hidden_dim]
        
        # Apply time dependent FiLM on projected theta with skip connections
        for i, blk in enumerate(self.theta_conditioned_blocks):
            gamma, beta = self.film_per_block[i](c)  # [batch_size, hidden_dim] for each
            h = blk(h, gamma, beta)  # [batch_size, hidden_dim]

        # Project theta back to original space
        return self.theta_tail(h)  # [batch_size, theta_dim]
    
    def set_noise_schedule(self, beta: torch.Tensor, eps: float = 1e-12):
        """Register noise schedule and precompute constants as model buffers.
        Expects beta shape [T]. Computes alpha, alpha_hat, sqrt terms, and posterior_std.
        
        Args:
            beta: Noise schedule, shape [num_timesteps]
            eps: Small constant for numerical stability
        """
        with torch.no_grad():
            beta = beta.detach()
            # Register primary schedule
            self.register_buffer('beta', beta)
            alpha = 1.0 - beta
            self.register_buffer('alpha', alpha)
            alpha_hat = torch.cumprod(alpha, dim=0)
            self.register_buffer('alpha_hat', alpha_hat)
            # Set number of timesteps from schedule length
            self.num_timesteps = beta.numel()

            # Precompute constants
            sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=eps))
            sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_hat, min=eps))
            self.register_buffer('sqrt_alpha', sqrt_alpha)
            self.register_buffer('sqrt_one_minus_alpha_bar', sqrt_one_minus_alpha_bar)

            # alpha_hat_{t-1} with alpha_hat_{-1} := 1, tm1 means time minus 1
            alpha_hat_tm1 = torch.cat([
                torch.ones(1, device=alpha_hat.device, dtype=alpha_hat.dtype),
                alpha_hat[:-1]
            ], dim=0)
            posterior_var = beta * (1.0 - alpha_hat_tm1) / torch.clamp(1.0 - alpha_hat, min=eps)
            posterior_var = torch.clamp(posterior_var, min=eps)
            posterior_std = torch.sqrt(posterior_var)
            self.register_buffer('posterior_std', posterior_std)
