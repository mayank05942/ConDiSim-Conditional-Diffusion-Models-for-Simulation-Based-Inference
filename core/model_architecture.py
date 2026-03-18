import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# Add the parent directory to the path so we can import from core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from train_utils
from core.train_utils import timestep_embedding, initialize_weights

class TimeEmbeddingModule(nn.Module):
    """Time embedding module that converts diffusion timesteps to feature vectors.
    
    Uses the timestep_embedding function from train_utils and projects it to the desired dimension.
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
        # Keep custom initialization only if you observe instability.
        # initialize_weights(self.mlp)

    def forward(self, t):
        """Convert timesteps to embeddings.
        
        Args:
            t: Tensor of shape [batch_size] containing timestep indices
            
        Returns:
            Tensor of shape [batch_size, code_dim] containing time embeddings
        """
        t_emb = timestep_embedding(t, self.sin_dim)
        return self.mlp(t_emb)  # [batch_size, code_dim]


class YFeatureEncoder(nn.Module):
    """Encoder for observation data.
    
    Encodes the observed data y into a feature vector.
    """
    def __init__(self, y_dim=100, hidden_dim=512, code_dim=64):
        super().__init__()
        # Encoder network: y_dim -> hidden_dim -> code_dim
        self.net = nn.Sequential(
            nn.Linear(y_dim, 4 * code_dim),  # 100 -> 256 -> 64
            nn.SiLU(),
            nn.Linear(4 * code_dim, code_dim),
        )
        self.initialize()
    
    def initialize(self):
        """Initialize weights using shared initialization function."""
        # Note: For this small model, PyTorch defaults are usually sufficient.
        # Keep custom initialization only if you observe instability.
        # initialize_weights(self.net)

    def forward(self, y):
        """Encode observation data.
        
        Args:
            y: Tensor of shape [batch_size, y_dim] containing observation data
            
        Returns:
            Tensor of shape [batch_size, code_dim] containing encoded features
        """
        return self.net(y)  # [batch_size, code_dim]

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
            #nn.Dropout(0.1),
            nn.Linear(h, hidden_dim),
        )
        with torch.no_grad():
            nn.init.zeros_(self.post_film_mlp[-1].weight)
            nn.init.zeros_(self.post_film_mlp[-1].bias)

        self.initialize()

    def initialize(self):
        """Initialize weights using shared initialization function."""
        # Note: For this small model, PyTorch defaults are usually sufficient.
        # Keep zero-initializing the last layer only if training stability benefits from it.
        # initialize_weights(self.post_film_mlp, zero_init_last=True, last_module=self.post_film_mlp[-1])

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


class ReverseDiffusionModel(nn.Module):
    """Reverse diffusion model for posterior sampling.
    
    Estimates the theta noise given y and t (reverse diffusion process).
    Supports Classifier-Free Guidance for improved sample quality.
    """
    def __init__(self, theta_dim, y_dim, hidden_dim: int = 128, code_dim: int | None = None, num_blocks: int = 6, cfg_dropout_prob: float = 0.2):
        """Initialize the reverse diffusion model.
        
        Args:
            theta_dim: Dimension of the parameter space
            y_dim: Dimension of the observation space
            cfg_dropout_prob: Probability of dropping condition during training for CFG
        """
        super().__init__()
        self.theta_dim = theta_dim  # dimension of theta
        self.y_dim = y_dim  # dimension of y
        self.cfg_dropout_prob = cfg_dropout_prob  # probability of dropping condition during training

        # Model hyperparameters
        if code_dim is None:
            code_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        self.num_blocks = num_blocks
        
        
        # Will be set by noise scheduler
        self.num_timesteps = 0
        
        # Time embedding
        self.timembedding = TimeEmbeddingModule(
            code_dim=self.code_dim,
            sin_dim=self.code_dim,
            hidden_dim=self.hidden_dim,
        )
        
        # Y encoder
        self.y_enc = YFeatureEncoder(
            y_dim=y_dim,
            hidden_dim=self.hidden_dim,
            code_dim=self.code_dim,
        )
        
        # Null embedding for unconditional generation (CFG)
        self.null_y_embedding = nn.Parameter(torch.zeros(1, self.code_dim))
        
        # Theta projection (use LayerNorm for stability regardless of input scaling)
        # Ensure output matches hidden_dim (64) expected by downstream blocks
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
                mlp_dim=self.hidden_dim,
            )
            for _ in range(self.num_blocks)
        ])

        # Output head (keep LayerNorm for stability; do not rely solely on input scaling)
        self.theta_tail = nn.Sequential(
            #nn.LayerNorm(self.hidden_dim),
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
        # Note: For this small model, PyTorch defaults are usually sufficient.
        # Keep custom/zero-last initialization only if instability is observed.
        # Initialize theta_head
        # initialize_weights(self.theta_head)
        
        # Initialize output head with zero-init for the last layer
        # initialize_weights(self.theta_tail, zero_init_last=True)

    def forward(self, theta_t, y, t, force_drop_condition=False):
        """Forward pass of the reverse diffusion model.
        
        Args:
            theta_t: Noisy parameters at timestep t, shape [batch_size, theta_dim]
            y: Observation data, shape [batch_size, y_dim]
            t: Timestep indices, shape [batch_size]
            force_drop_condition: If True, always use unconditional path (for CFG)
            
        Returns:
            Predicted noise, shape [batch_size, theta_dim]
        """
        batch_size = theta_t.shape[0]
        device = theta_t.device
        
        # Encode time
        t_code = self.timembedding(t)  # [batch_size, code_dim]
        
        # # Determine if we should drop the condition for this batch (for CFG training)
        # if force_drop_condition:
        #     # Use null embedding for unconditional generation
        #     y_code = self.null_y_embedding.expand(batch_size, -1)  # [batch_size, code_dim]
        # else:
        #     # Encode observation
        #     y_code = self.y_enc(y)  # [batch_size, code_dim]
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

        # Project theta back to original space - removed out_norm layer
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


if __name__ == "__main__":
    """Test the model architecture."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test parameters
    batch_size = 16
    theta_dim = 10
    y_dim = 10
    
    # Create model
    model = ReverseDiffusionModel(theta_dim=theta_dim, y_dim=y_dim).to(device)
    
    # Print model summary
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    theta_t = torch.randn(batch_size, theta_dim, device=device)
    y = torch.randn(batch_size, y_dim, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    
    try:
        # Forward pass
        print("\nTesting forward pass...")
        output = model(theta_t, y, t)
        print(f"Input shapes: theta_t={theta_t.shape}, y={y.shape}, t={t.shape}")
        print(f"Output shape: {output.shape}")
        print("Forward pass successful!")
        
        # Test with noise scheduler
        print("\nNote: To test sampling, you need to register a noise scheduler first.")
        print("Example usage:")
        print("  from core.noise_scheduler import NoiseScheduler")
        print("  scheduler = NoiseScheduler(num_timesteps=1000, beta_schedule='quadratic')")
        print("  scheduler.register_to_model(model)")
        print("  samples = model.sample(N=100, y_observed=y_observed)")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    print("\nModel architecture test complete!")
