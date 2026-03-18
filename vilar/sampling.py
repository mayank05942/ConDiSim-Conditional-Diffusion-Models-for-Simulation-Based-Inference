import torch
import numpy as np
import time
import os

def sample_posterior(model, y_observed, num_samples=1000, device=None, lambda_guidance=3.0):
    """
    Sample from the posterior distribution using the trained diffusion model.
    Supports Classifier-Free Guidance for improved sample quality.
    
    Args:
        model: Trained diffusion model
        y_observed: Observed data, shape [1, y_channels, y_seq_length] or [batch_size, y_channels, y_seq_length]
        num_samples: Number of posterior samples to generate
        device: Device to use for sampling
        lambda_guidance: CFG guidance strength parameter (0.0 = no guidance, >0.0 = stronger guidance)
        
    Returns:
        Posterior samples, shape [num_samples, theta_dim]
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Check if y_observed needs to be repeated
    if y_observed.shape[0] == 1:
        y_observed = y_observed.repeat(num_samples, 1, 1)
    elif y_observed.shape[0] != num_samples:
        raise ValueError(f"y_observed batch size ({y_observed.shape[0]}) must be 1 or equal to num_samples ({num_samples})")
    
    # Move to device
    y_observed = y_observed.to(device)
    
    # Determine if we should use CFG
    use_cfg = lambda_guidance > 0.0 and hasattr(model, 'null_y_embedding')
    
    # Measure sampling time
    start_time = time.time()
    
    with torch.no_grad():
        # Start with random noise
        theta = torch.randn((num_samples, model.theta_dim), device=device)
        T = model.num_timesteps

        # Reverse diffusion process (denoising)
        for t in reversed(range(T)):
            t_tensor = torch.full((num_samples,), t, device=device)
            beta_t = model.beta[t].unsqueeze(-1)
            sqrt_one_minus_alpha_bar_t = model.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
            sqrt_alpha_t = model.sqrt_alpha[t].unsqueeze(-1)

            # Predict noise with CFG if enabled
            if use_cfg:
                # Get conditional prediction
                eps_pred_cond = model(theta, y_observed, t_tensor, force_drop_condition=False)
                
                # Get unconditional prediction
                eps_pred_uncond = model(theta, y_observed, t_tensor, force_drop_condition=True)
                
                # Apply classifier-free guidance formula:
                # ε = ε_cond + λ(ε_cond - ε_uncond)
                # where λ is our lambda_guidance parameter
                eps_pred = eps_pred_cond + lambda_guidance * (eps_pred_cond - eps_pred_uncond)
            else:
                # Standard prediction without guidance
                eps_pred = model(theta, y_observed, t_tensor)

            # DDPM denoising step using precomputed constants
            mean = (theta - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_pred) / sqrt_alpha_t

            if t > 0:
                noise = torch.randn_like(theta)
                posterior_std_t = model.posterior_std[t].unsqueeze(-1)
                theta = mean + posterior_std_t * noise
            else:
                theta = mean
    
    end_time = time.time()
    sampling_time = end_time - start_time
    print(f"Sampling completed in {sampling_time:.2f} seconds")
    
    return theta


def save_posterior_samples(samples, save_dir="posterior_samples", filename=None):
    """
    Save posterior samples to a file.
    
    Args:
        samples: Posterior samples, shape [num_samples, theta_dim] or dict of arrays
        save_dir: Directory to save samples
        filename: Filename for the samples
        
    Returns:
        Path to the saved samples
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Generate filename if not provided
    if filename is None:
        filename = f"posterior_samples_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    
    # Full path to save file
    save_path = os.path.join(save_dir, filename)
    
    # Save the samples
    if isinstance(samples, dict):
        np.savez(save_path, **samples)
    else:
        np.savez(save_path, posterior_samples=samples)
    
    print(f"Posterior samples saved to {save_path}")
    
    return save_path


def sample_and_save(model, y_observed, theta_scaler, save_dir="posterior_samples", filename=None, num_samples=1000, device=None, lambda_guidance=2.0, extra_arrays=None):
    """
    Sample from the posterior and save the samples.
    Supports Classifier-Free Guidance for improved sample quality.
    
    Args:
        model: Trained diffusion model
        y_observed: Observed data, shape [1, y_channels, y_seq_length] or [batch_size, y_channels, y_seq_length]
        theta_scaler: StandardScaler for theta parameters
        save_dir: Directory to save samples
        filename: Filename for the samples
        num_samples: Number of posterior samples to generate
        device: Device to use for sampling
        lambda_guidance: CFG guidance strength parameter (0.0 = no guidance, >0.0 = stronger guidance)
        extra_arrays: Additional arrays to save with the samples
        
    Returns:
        Tuple of (descaled samples, sampling time)
    """
    # Measure sampling time
    start_time = time.time()
    
    # Sample from posterior with classifier-free guidance
    theta_samples = sample_posterior(model, y_observed, num_samples, device, lambda_guidance=lambda_guidance)
    
    # Descale samples
    theta_samples_descaled = theta_scaler.inverse_transform(theta_samples.cpu().numpy())
    
    # Calculate sampling time
    end_time = time.time()
    sampling_time = end_time - start_time
    
    # Save samples
    arrays_to_save = {"theta_samples": theta_samples_descaled}
    
    # Add extra arrays if provided
    if extra_arrays:
        for key, value in extra_arrays.items():
            if value is None:
                continue
            arrays_to_save[key] = np.asarray(value)
    
    # Save the samples
    save_path = save_posterior_samples(arrays_to_save, save_dir, filename)
    
    return theta_samples_descaled, sampling_time
