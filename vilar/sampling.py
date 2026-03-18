import torch
import numpy as np
import time
import os

def sample_posterior(model, y_observed, num_samples=1000, device=None, lambda_guidance=3.0):
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    if y_observed.shape[0] == 1:
        y_observed = y_observed.repeat(num_samples, 1, 1)
    elif y_observed.shape[0] != num_samples:
        raise ValueError(f"y_observed batch size ({y_observed.shape[0]}) must be 1 or equal to num_samples ({num_samples})")
    
    y_observed = y_observed.to(device)
    
    use_cfg = lambda_guidance > 0.0 and hasattr(model, 'null_y_embedding')
    
    start_time = time.time()
    
    with torch.no_grad():
        theta = torch.randn((num_samples, model.theta_dim), device=device)
        T = model.num_timesteps

        for t in reversed(range(T)):
            t_tensor = torch.full((num_samples,), t, device=device)
            beta_t = model.beta[t].unsqueeze(-1)
            sqrt_one_minus_alpha_bar_t = model.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
            sqrt_alpha_t = model.sqrt_alpha[t].unsqueeze(-1)

            if use_cfg:
                eps_pred_cond = model(theta, y_observed, t_tensor, force_drop_condition=False)
                
                eps_pred_uncond = model(theta, y_observed, t_tensor, force_drop_condition=True)
                
                eps_pred = eps_pred_cond + lambda_guidance * (eps_pred_cond - eps_pred_uncond)
            else:
                eps_pred = model(theta, y_observed, t_tensor)

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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if filename is None:
        filename = f"posterior_samples_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    
    save_path = os.path.join(save_dir, filename)
    
    if isinstance(samples, dict):
        np.savez(save_path, **samples)
    else:
        np.savez(save_path, posterior_samples=samples)
    
    print(f"Posterior samples saved to {save_path}")
    
    return save_path


def sample_and_save(model, y_observed, theta_scaler, save_dir="posterior_samples", filename=None, num_samples=1000, device=None, lambda_guidance=2.0, extra_arrays=None):
    start_time = time.time()
    
    theta_samples = sample_posterior(model, y_observed, num_samples, device, lambda_guidance=lambda_guidance)
    
    theta_samples_descaled = theta_scaler.inverse_transform(theta_samples.cpu().numpy())
    
    end_time = time.time()
    sampling_time = end_time - start_time
    
    arrays_to_save = {"theta_samples": theta_samples_descaled}
    
    if extra_arrays:
        for key, value in extra_arrays.items():
            if value is None:
                continue
            arrays_to_save[key] = np.asarray(value)
    
    save_path = save_posterior_samples(arrays_to_save, save_dir, filename)
    
    return theta_samples_descaled, sampling_time
