import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from vilar_model_architecture import DiffusionModel
from train_utils import set_seed, print_network_architecture, make_yolox_warmcos_scheduler
from noise_scheduler import NoiseScheduler


def diffusion_loss(model,
                   theta_0,
                   y,
                   num_timesteps,
                   alpha_hat,
                   use_snr_weighting: bool = True,
                   eps: float = 1e-12,
                   loss_type: str = "mse",
                   huber_beta: float = 1.0
                   ):
    """
    Compute diffusion loss (predicting noise) with optional Min-SNR gamma weighting.
    Supports Classifier-Free Guidance training by randomly dropping conditions.

    Args:
        model: The diffusion model
        theta_0: Clean parameters, shape [batch_size, theta_dim]
        y: Observation data, shape [batch_size, y_dim]
        num_timesteps: Total number of diffusion timesteps
        alpha_hat: Cumulative product of (1-beta), shape [num_timesteps]
        use_snr_weighting: Whether to use SNR-based loss weighting
        eps: Small constant for numerical stability
        loss_type: "mse" or "huber" (smooth L1 with beta=huber_beta)
        huber_beta: Beta parameter for Huber loss
       

    Returns:
        Diffusion loss (scalar)
    """
    # Sample timesteps and construct noisy theta_t
    device = theta_0.device
    batch_size = theta_0.size(0)
    
    t = torch.randint(0, num_timesteps, (batch_size,), device=device).long()
    noise = torch.randn_like(theta_0)
    alpha_hat = alpha_hat.to(device)
    abar_t = alpha_hat[t].to(device)  # shape [B]
    
    # Construct noisy theta_t 
    sqrt_abar = torch.sqrt(abar_t.unsqueeze(-1))
    sqrt_1m_abar = torch.sqrt(torch.clamp(1 - abar_t, min=eps).unsqueeze(-1))
    theta_t = sqrt_abar * theta_0 + sqrt_1m_abar * noise
    
    
    # Ensure gradients are enabled for model forward pass
    with torch.enable_grad():
        pred_noise = model(theta_t, y, t)

    if not use_snr_weighting:
        if loss_type == "mse":
            return F.mse_loss(pred_noise.float(), noise.float(), reduction="mean")
        elif loss_type == "huber":
            return F.smooth_l1_loss(pred_noise.float(), noise.float(), beta=huber_beta, reduction='mean')
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    # Per-sample error for weighting
    if loss_type == "mse":
        loss = F.mse_loss(pred_noise.float(), noise.float(), reduction="none")  # [B, D]
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(pred_noise.float(), noise.float(), beta=huber_beta, reduction='none')  # [B, D]
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    loss = loss.mean(dim=1)  # [B]

    # Compute SNR_t = alpha_bar_t / (1 - alpha_bar_t)
    snr = torch.clamp(abar_t, min=eps) / torch.clamp(1.0 - abar_t, min=eps)  # [B]
    gamma = 5.0
    base_weight = torch.minimum(snr, torch.full_like(snr, gamma)) / snr

    loss = (loss * base_weight).mean()
    return loss


# Set up argument parser
parser = argparse.ArgumentParser(description='Train diffusion model for Vilar parameter inference')
parser.add_argument('--budget', type=int, default=10000, help='Dataset budget (10000, 20000, or 30000)')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--num_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
parser.add_argument('--beta_schedule', type=str, default='quadratic', help='Beta schedule (linear or quadratic)')
parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
parser.add_argument('--code_dim', type=int, default=128, help='Code dimension')
parser.add_argument('--num_blocks', type=int, default=6, help='Number of residual blocks')
parser.add_argument('--use_snr_weighting', type=int, default=0, help='Use SNR weighting (0 or 1)')
parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

# Parse arguments from environment variables if available
args = parser.parse_args()

# Set random seed for reproducibility
set_seed(args.seed)

# Create save directory if it doesn't exist
os.makedirs(args.save_dir, exist_ok=True)

# Device
device = torch.device(args.device)
print(f"Using device: {device}")

# Load dataset
def load_dataset(budget):
    dataset_path = f'/cephyr/users/nautiyal/Alvis/diffusion/vilar/datasets/vilar_dataset_{budget}.npz'
    print(f"Loading dataset: {dataset_path}")
    
    data = np.load(dataset_path, allow_pickle=True)
    
    # Extract data
    theta = data['theta_norm']  # Use normalized parameters
    ts_data = data['ts_data_norm']  # Use normalized time series
    
    # Convert to tensors
    theta = torch.tensor(theta, dtype=torch.float32)
    ts_data = torch.tensor(ts_data, dtype=torch.float32)
    
    print(f"Loaded {len(theta)} samples")
    print(f"Theta shape: {theta.shape}")
    print(f"Time series shape: {ts_data.shape}")
    
    return theta, ts_data

# Load data
theta, ts_data = load_dataset(args.budget)

# Create dataset and split into train/val with 30% validation split
dataset = TensorDataset(theta, ts_data)
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Create model
model = DiffusionModel(
    theta_dim=15,
    y_channels=3,
    y_seq_length=200,
    cfg_dropout_prob=0.2  # Use the default value from core implementation
).to(device)

# Set number of timesteps
model.num_timesteps = args.num_timesteps

# Print model architecture
print_network_architecture(model)

# Create optimizer
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

# Calculate total steps for YOLOX scheduler
total_steps = args.num_epochs * len(train_loader)

# Create YOLOX warmcos scheduler from core implementation
scheduler = make_yolox_warmcos_scheduler(
    optimizer,
    total_steps,
    base_lr=args.learning_rate,
    warmup_ratio=0.10,  # 10% of steps for warmup
    warmup_lr_ratio=0.20  # Start at 20% of base_lr
)

# Create noise scheduler from core implementation
noise_scheduler = NoiseScheduler(
    num_timesteps=args.num_timesteps,
    beta_schedule=args.beta_schedule,
    device=device
)

# Register noise schedule to model
noise_scheduler.register_to_model(model)

# Training function
def train_epoch(model, train_loader, optimizer, scheduler, use_snr_weighting):
    model.train()
    epoch_train_loss = 0.0
    
    for theta_batch, ts_batch in tqdm(train_loader, desc="Training"):
        theta_batch = theta_batch.to(device)
        ts_batch = ts_batch.to(device)
        
        # Use the diffusion_loss function from the core implementation
        optimizer.zero_grad()
        loss = diffusion_loss(
            model,
            theta_batch,
            ts_batch,
            args.num_timesteps,
            model.alpha_hat,
            use_snr_weighting=use_snr_weighting
        )
        loss.backward()
        
        # Gradient clipping - increased to 5.0 based on recommendations
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        scheduler.step()
        
        epoch_train_loss += loss.item()
    
    return epoch_train_loss / len(train_loader)

# Validation function
def validate(model, val_loader, use_snr_weighting):
    model.eval()
    epoch_val_loss = 0.0
    
    with torch.no_grad():
        for theta_batch, ts_batch in tqdm(val_loader, desc="Validation"):
            theta_batch = theta_batch.to(device)
            ts_batch = ts_batch.to(device)
            
            # Use the diffusion_loss function from the core implementation
            loss = diffusion_loss(
                model,
                theta_batch,
                ts_batch,
                args.num_timesteps,
                model.alpha_hat,
                use_snr_weighting=use_snr_weighting
            )
            
            epoch_val_loss += loss.item()
    
    return epoch_val_loss / len(val_loader)

# Training loop
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0
patience = 15  # Early stopping patience (reduced from 20 to 15 based on recommendations)

print(f"Starting training for {args.num_epochs} epochs")
start_time = time.time()

for epoch in range(args.num_epochs):
    # Train
    train_loss = train_epoch(
        model, train_loader, optimizer, scheduler, args.use_snr_weighting
    )
    train_losses.append(train_loss)
    
    # Validate
    val_loss = validate(
        model, val_loader, args.use_snr_weighting
    )
    val_losses.append(val_loss)
    
    # Print progress
    print(f"Epoch {epoch+1}/{args.num_epochs}, "
          f"Train Loss: {train_loss:.6f}, "
          f"Val Loss: {val_loss:.6f}, "
          f"Time: {time.time() - start_time:.2f}s")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        
        # Save model
        model_path = os.path.join(
            args.save_dir, 
            f"vilar_diffusion_budget{args.budget}_best.pt"
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'args': vars(args)
        }, model_path)
        print(f"Saved best model to {model_path}")
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= patience:
        print(f"Early stopping after {epoch+1} epochs")
        break

# Save final model
model_path = os.path.join(
    args.save_dir, 
    f"vilar_diffusion_budget{args.budget}_final.pt"
)
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'args': vars(args)
}, model_path)
print(f"Saved final model to {model_path}")

# Plot training curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss (Budget: {args.budget})')
plt.legend()
plt.grid(True)

# Save plot
plot_path = os.path.join(
    args.save_dir, 
    f"vilar_diffusion_budget{args.budget}_loss.png"
)
plt.savefig(plot_path)
print(f"Saved loss plot to {plot_path}")

print("Training complete!")
