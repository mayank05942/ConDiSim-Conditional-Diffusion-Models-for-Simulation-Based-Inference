import torch
import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import LambdaLR
import random


def timestep_embedding(timesteps,
                       dim,
                       max_period=10000):
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


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_yolox_warmcos_scheduler(
    optimizer,
    total_steps: int,
    *,
    base_lr: float | None = None,     # if None, read from optimizer
    min_lr: float | None = None,      # if None, 1% of base_lr
    warmup_ratio: float = 0.10,       # first % of steps for warmup (increased from 0.05)
    warmup_lr_ratio: float = 0.20,    # warmup starts at this * base_lr (increased from 0.10)
    no_aug_ratio: float = 0.05        # last % of steps flat at min_lr
):
    """
    YOLOX-style learning rate schedule for diffusion models:
      - Quadratic warmup from warmup_lr_start -> base_lr
      - Cosine decay from base_lr -> min_lr
      - Flat tail at min_lr during last no_aug steps
    
    Args:
        optimizer: PyTorch optimizer
        total_steps: Total number of training steps (epochs * batches_per_epoch)
        base_lr: Base learning rate (if None, read from optimizer)
        min_lr: Minimum learning rate (if None, 1% of base_lr)
        warmup_ratio: Fraction of steps for warmup phase
        warmup_lr_ratio: Initial LR during warmup as fraction of base_lr
        no_aug_ratio: Fraction of steps for final flat phase at min_lr
    
    Returns:
        LambdaLR scheduler
    """
    assert total_steps > 0, "total_steps must be > 0"
    warmup_steps = max(1, int(round(warmup_ratio * total_steps)))
    no_aug_steps = max(1, int(round(no_aug_ratio * total_steps)))
    main_steps = max(1, total_steps - warmup_steps - no_aug_steps)

    # Capture initial per-group base LRs (allows different LRs per param group)
    init_lrs = [pg['lr'] for pg in optimizer.param_groups]
    if base_lr is not None:
        init_lrs = [base_lr for _ in init_lrs]
    min_lrs = [
        (min_lr if min_lr is not None else max(1e-6, 0.01 * blr))
        for blr in init_lrs
    ]
    warmup_starts = [max(1e-8, warmup_lr_ratio * blr) for blr in init_lrs]

    def lr_lambda_factory(blr, minlr, wstart):
        def lr_lambda(step: int):
            # step is 0-based in LambdaLR
            if step < warmup_steps:
                # Quadratic warmup: from wstart -> blr
                u = (step + 1) / float(warmup_steps)
                return (wstart + (blr - wstart) * (u ** 2)) / blr
            elif step >= warmup_steps + main_steps:
                # Flat tail (no-aug) at minlr
                return minlr / blr
            else:
                # Cosine phase
                k = step - warmup_steps
                u = k / float(main_steps)
                cos_val = 0.5 * (1.0 + math.cos(math.pi * u))
                lr_now = minlr + (blr - minlr) * cos_val
                return lr_now / blr
        return lr_lambda

    lambdas = [lr_lambda_factory(blr, mlr, wstart) for blr, mlr, wstart in zip(init_lrs, min_lrs, warmup_starts)]
    scheduler = LambdaLR(optimizer, lr_lambda=lambdas)
    scheduler.total_steps = total_steps  # (optional) for reference
    scheduler.warmup_steps = warmup_steps
    scheduler.no_aug_steps = no_aug_steps
    scheduler.main_steps = main_steps
    return scheduler


def initialize_weights(module, zero_init_last=False, last_module=None):
    """Initialize weights using Kaiming initialization.
    
    Args:
        module: PyTorch module to initialize
        zero_init_last: Whether to zero-initialize the last layer
        last_module: The last module to zero-initialize (if zero_init_last is True)
    """
    last_linear = None
    
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            torch.nn.init.zeros_(m.bias)
            last_linear = m
    
    # Zero-initialize the last layer if specified
    if zero_init_last and last_linear is not None:
        if last_module is not None and last_linear is last_module:
            torch.nn.init.zeros_(last_linear.weight)
            torch.nn.init.zeros_(last_linear.bias)
        elif last_module is None:
            torch.nn.init.zeros_(last_linear.weight)
            torch.nn.init.zeros_(last_linear.bias)


def print_network_architecture(model):
    """Print detailed information about the network architecture.
    
    Args:
        model: The PyTorch model to analyze
    """
    print("\n" + "=" * 50)
    print("NETWORK ARCHITECTURE DETAILS")
    print("=" * 50)
    
    # Print model summary
    print(f"\nModel Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter Counts:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Print hyperparameters if available
    if hasattr(model, 'hidden_dim'):
        print(f"\nModel Hyperparameters:")
        print(f"  Hidden dimension: {model.hidden_dim}")
    if hasattr(model, 'code_dim'):
        print(f"  Code dimension: {model.code_dim}")
    if hasattr(model, 'num_blocks'):
        print(f"  Number of blocks: {model.num_blocks}")
    if hasattr(model, 'dropout'):
        print(f"  Dropout rate: {model.dropout}")
    if hasattr(model, 'num_timesteps'):
        print(f"  Number of timesteps: {model.num_timesteps}")
    
    # Print module-specific information
    print(f"\nModule Structure:")
    
    # Check for specific modules
    if hasattr(model, 'timembedding'):
        print(f"  Time Embedding: {model.timembedding.__class__.__name__}")
        if hasattr(model.timembedding, 'code_dim'):
            print(f"    - Output dimension: {model.timembedding.code_dim}")
    
    if hasattr(model, 'y_enc'):
        print(f"  Observation Encoder: {model.y_enc.__class__.__name__}")
        if hasattr(model.y_enc, 'net'):
            print(f"    - Structure: {model.y_enc.net}")
    
    if hasattr(model, 'film_per_block'):
        print(f"  FiLM Conditioning: {len(model.film_per_block)} blocks")
    
    if hasattr(model, 'theta_conditioned_blocks'):
        print(f"  Diffusion Blocks: {len(model.theta_conditioned_blocks)}")
        if len(model.theta_conditioned_blocks) > 0:
            block = model.theta_conditioned_blocks[0]
            if hasattr(block, 'post_film_mlp'):
                print(f"    - Block MLP: {block.post_film_mlp}")
    
    print("\n" + "=" * 50)
