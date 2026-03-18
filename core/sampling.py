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
        y_observed: Observed data, shape [1, y_dim] or [batch_size, y_dim]
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
        y_observed = y_observed.repeat(num_samples, 1)
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


def save_posterior_samples(samples, task_name, budget, run, results_dir="results"):
    """
    Save posterior samples to a file.
    
    Args:
        samples: Posterior samples, shape [num_samples, theta_dim] or dict of arrays
        task_name: Name of the task
        budget: Simulation budget
        run: Run number
        results_dir: Directory to save results
        
    Returns:
        Path to the saved samples
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
    
    # Use the same filename format as in hh_main.py for HH task
    if task_name == "hh":
        posterior_filename = f"hh_posterior_samples_budget_{budget}_run_{run}.npz"
    else:
        posterior_filename = f"{task_name}_run_{run}_budget_{budget}.npz"
        
    posterior_path = os.path.join(task_results_dir, posterior_filename)

    # Save the data with the same key names as in hh_main.py for HH task
    if task_name == "hh" and isinstance(samples, dict):
        # Rename keys to match hh_main.py format
        save_dict = {
            "theta_samples": samples["theta_samples"],
            "true_parameters": samples.get("true_parameters", None),
            "observed_data": samples.get("observed_data", None),
            "true_V": samples.get("voltage_trace", None),
            "true_H": samples.get("energy_trace", None),
            "budget": budget,
            "run": run
        }
        np.savez(posterior_path, **save_dict)
    elif isinstance(samples, dict):
        np.savez(posterior_path, **samples)
    else:
        np.savez(posterior_path, theta_samples=samples)
        
    print(f"Posterior samples saved to {posterior_path}")
    
    return posterior_path


def sample_and_save(model, y_observed, theta_scaler, task_name, budget, run, num_samples=1000, device=None, results_dir="results", lambda_guidance=2.0, extra_arrays=None):
    """
    Sample from the posterior and save the samples.
    Supports Classifier-Free Guidance for improved sample quality.
    
    Args:
        model: Trained diffusion model
        y_observed: Observed data, shape [1, y_dim] or [batch_size, y_dim]
        theta_scaler: StandardScaler for theta parameters
        task_name: Name of the task
        budget: Simulation budget
        run: Run number
        num_samples: Number of posterior samples to generate
        device: Device to use for sampling
        results_dir: Directory to save results
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
            
        # For HH task, rename keys to match hh_main.py format
        if task_name == "hh":
            # Map the keys from extra_arrays to the keys used in hh_main.py
            key_mapping = {
                "true_parameters": "true_parameters",
                "observed_data": "observed_data",
                "voltage_trace": "true_V",
                "energy_trace": "true_H"
            }
            
            # Add budget and run information
            arrays_to_save["budget"] = budget
            arrays_to_save["run"] = run

    # Save the samples
    save_path = save_posterior_samples(arrays_to_save, task_name, budget, run, results_dir)
    
    # Log sampling time
    time_log_dir = os.path.join(results_dir, task_name)
    # Ensure the directory exists
    if not os.path.exists(time_log_dir):
        os.makedirs(time_log_dir)
        
    time_log_path = os.path.join(time_log_dir, "sampling_times.txt")
    with open(time_log_path, 'a') as f:
        f.write(f"Task: {task_name}\n")
        f.write(f"Simulation Budget: {budget}\n")
        f.write(f"Run: {run}\n")
        f.write(f"Sampling time: {sampling_time:.2f} seconds\n\n")
    
    return theta_samples_descaled, sampling_time


if __name__ == "__main__":
    print("This module provides sampling functionality for diffusion models.")
    print("Import and use the functions in your main script.")
