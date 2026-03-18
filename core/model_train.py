import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import time
import os
from torch.utils.data import DataLoader

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


def train_model(model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                num_epochs,
                num_timesteps,
                alpha_hat,
                patience=30,  # Increased patience for early stopping
                min_epochs_before_es=50,  # Ensure at least 50 epochs of training
                use_snr_weighting: bool = True,  # SNR weighting improves training stability
                max_grad_norm: float = 5.0,  # Increased gradient clipping for better exploration
                cfg_training: bool = True,  # Enable Classifier-Free Guidance training
                device=None):
    """
    Train the diffusion model.

    Args:
        model: The diffusion model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        num_epochs: Maximum number of training epochs
        num_timesteps: Number of diffusion timesteps
        alpha_hat: Cumulative product of (1-beta), shape [num_timesteps]
        patience: Number of epochs to wait for improvement before early stopping
        min_epochs_before_es: Minimum number of epochs before early stopping
        use_snr_weighting: Whether to use SNR-based loss weighting
        max_grad_norm: Maximum gradient norm for gradient clipping
        device: Device to use for training

    Returns:
        Tuple of (trained model, training losses, validation losses)
    """
    if device is None:
        device = next(model.parameters()).device
    
    best_val_loss = float('inf')
    best_model = None
    counter = 0

    train_losses = []
    val_losses = []
    
    # Create YOLOX scheduler if it wasn't provided
    if scheduler is None:
        from train_utils import make_yolox_warmcos_scheduler
        total_steps = num_epochs * len(train_loader)
        scheduler = make_yolox_warmcos_scheduler(
            optimizer,
            total_steps=total_steps,
            min_lr=1e-6,
            warmup_ratio=0.05,
            warmup_lr_ratio=0.10,
            no_aug_ratio=0.05
        )
        print(f"Created YOLOX scheduler with {total_steps} total steps")
        print(f"Warmup: {scheduler.warmup_steps} steps, Main phase: {scheduler.main_steps} steps, Final phase: {scheduler.no_aug_steps} steps")

    # Determine if we're using a per-batch scheduler
    is_batch_scheduler = not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        
        # Enable gradients for the entire training loop
        with torch.enable_grad():
            for theta_batch, y_batch in train_loader:
                theta_batch = theta_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                loss = diffusion_loss(
                    model,
                    theta_batch,
                    y_batch,
                    num_timesteps,
                    alpha_hat,
                    use_snr_weighting=use_snr_weighting,
                )
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                
                # Step the scheduler after each batch for YOLOX scheduler
                if is_batch_scheduler:
                    scheduler.step()

                epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for theta_batch, y_batch in val_loader:
                theta_batch = theta_batch.to(device)
                y_batch = y_batch.to(device)
                val_loss += diffusion_loss(
                    model,
                    theta_batch,
                    y_batch,
                    num_timesteps,
                    alpha_hat,
                    use_snr_weighting=use_snr_weighting,
                ).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}, Learning Rate: {current_lr:.6f}")

        # For ReduceLROnPlateau, step with validation loss after each epoch
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            counter = 0
        else:
            # Only start counting patience after min_epochs_before_es
            if (epoch + 1) >= min_epochs_before_es:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered.")
                    model = best_model
                    break
    
    return model, train_losses, val_losses


def create_optimizer_and_scheduler(model, learning_rate=1e-3, weight_decay=1e-4, num_epochs=200, steps_per_epoch=None):
    """
    Create optimizer and learning rate scheduler.

    Args:
        model: The model to optimize
        learning_rate: Initial learning rate
        weight_decay: Weight decay factor
        num_epochs: Number of training epochs
        steps_per_epoch: Number of steps per epoch (if None, will be set later)

    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # We'll use the YOLOX warmup-cosine scheduler for better diffusion training
    # The actual scheduler will be created in train_model when we know steps_per_epoch
    if steps_per_epoch is None:
        # Return a placeholder - we'll create the real scheduler in train_model
        scheduler = None
    else:
        # If steps_per_epoch is provided, we can create the scheduler now
        from train_utils import make_yolox_warmcos_scheduler
        total_steps = num_epochs * steps_per_epoch
        scheduler = make_yolox_warmcos_scheduler(
            optimizer,
            total_steps=total_steps,
            base_lr=learning_rate,
            min_lr=5e-7,
            warmup_ratio=0.02,  # 5% of steps for warmup
            warmup_lr_ratio=0.05,  # Start at 10% of base LR
            no_aug_ratio=0.01  # 5% of steps at min_lr
        )
    
    return optimizer, scheduler


def save_model(model, train_losses, val_losses, training_time, task_name, budget, run, results_dir="results"):
    """
    Save the trained model and training information.

    Args:
        model: The trained model
        train_losses: List of training losses
        val_losses: List of validation losses
        training_time: Total training time in seconds
        task_name: Name of the task
        budget: Simulation budget
        run: Run number
        results_dir: Directory to save results

    Returns:
        Path to the saved model
    """
    # Use absolute path for results directory
    base_dir = "/cephyr/users/nautiyal/Alvis/diffusion"
    
    # If results_dir is not an absolute path, make it absolute
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(base_dir, results_dir)
    
    # Create results folder structure
    task_results_dir = os.path.join(results_dir, task_name)
    if not os.path.exists(task_results_dir):
        os.makedirs(task_results_dir)
    
    training_info = {
        'model_state_dict': model.state_dict(),
        'training_time': training_time,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    # Save the model and training info
    model_filename = f"{task_name}_budget_{budget}_run_{run}.pth"
    model_path = os.path.join(task_results_dir, model_filename)
    torch.save(training_info, model_path)
    print(f"Model saved to {model_path}")
    
    # Convert training time to minutes
    training_time_minutes = training_time / 60
    
    # Save training time to txt file
    time_log_path = os.path.join(task_results_dir, "training_times.txt")
    with open(time_log_path, 'a') as f:
        f.write(f"Task: {task_name}\n")
        f.write(f"Simulation Budget: {budget}\n")
        f.write(f"Run: {run}\n")
        f.write(f"Training Time (minutes): {training_time_minutes:.2f}\n")
        f.write("-" * 50 + "\n")
    
    print(f"Training time: {training_time_minutes/60:.2f} hours ({training_time_minutes:.2f} minutes)")
    
    return model_path


if __name__ == "__main__":
    print("This module provides training functionality for diffusion models.")
    print("Import and use the functions in your main script.")
