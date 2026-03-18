import torch
import math
import numpy as np
import matplotlib.pyplot as plt


class NoiseScheduler:
    """A class that encapsulates noise scheduling for diffusion models.
    
    This class handles different noise schedule types (linear, cosine, quadratic) and
    computes all necessary diffusion constants (alpha, alpha_hat, etc.).
    """
    
    def __init__(self, 
                 num_timesteps: int = 1000,
                 beta_schedule: str = "linear",
                 max_beta: float = 0.999,
                 eps: float = 1e-12,
                 device=None,
                 dtype=torch.float32):
        """Initialize the noise scheduler.
        
        Args:
            num_timesteps: Number of diffusion timesteps
            beta_schedule: Type of schedule ('linear', 'quadratic', or 'cosine')
            max_beta: Maximum beta value (for clamping)
            eps: Small constant for numerical stability
            device: Computation device
            dtype: Data type for computations
        """
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.max_beta = max_beta
        self.eps = eps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        # Generate beta schedule and compute derived quantities
        self.beta = self._get_beta_schedule()
        self._compute_diffusion_constants()
    
    def _betas_for_alpha_bar(self, alpha_bar_fn):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].
        
        Args:
            alpha_bar_fn: a function that takes an argument t from 0 to 1 and
                          produces the cumulative product of (1-beta) up to that
                          part of the diffusion process.
        """
        betas = []
        for i in range(self.num_timesteps):
            t1 = i / self.num_timesteps
            t2 = (i + 1) / self.num_timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), self.max_beta))
        return torch.tensor(betas, device=self.device, dtype=self.dtype)
    
    def _cosine_beta_schedule(self, s: float = 0.008):
        """Compute betas for the cosine scheduler (Nichol & Dhariwal 2021, improved DDPM).
        
        This implementation has improved numerical stability and better scaling.
        """
        # Direct computation of betas for better numerical stability
        steps = self.num_timesteps
        t = torch.linspace(0, 1, steps + 1, device=self.device, dtype=self.dtype)
        alpha_bar = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = torch.clamp(alpha_bar, min=self.eps)
        
        # Compute betas directly from alpha_bar values
        alpha_bar = alpha_bar / alpha_bar[0]  # Normalize to ensure alpha_bar[0] = 1
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        
        # Scale betas based on timesteps similar to linear/quadratic schedules
        scale_factor = 1000 / self.num_timesteps
        betas = betas * scale_factor * 0.008  # Apply scaling similar to other schedules
        
        return torch.clamp(betas, min=self.eps, max=self.max_beta)
    
    def _get_beta_schedule(self):
        """Generate the beta schedule based on the specified type."""
        if self.beta_schedule == "linear":
            # Linear schedule from Ho et al, extended to work for any number of diffusion steps
            scale = 1000 / self.num_timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            betas = torch.linspace(beta_start, beta_end, self.num_timesteps, device=self.device, dtype=self.dtype)
            
        elif self.beta_schedule == "quadratic":
            # Quadratic schedule with scaled values based on the linear schedule
            scale = 1000 / self.num_timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            u = torch.linspace(0.0, 1.0, self.num_timesteps, device=self.device, dtype=self.dtype)
            betas = beta_start + (beta_end - beta_start) * (u ** 2)
            
        elif self.beta_schedule == "cosine":
            # Cosine schedule doesn't use beta_start/beta_end - it generates its own values
            betas = self._cosine_beta_schedule()
            
        else:
            raise ValueError(f"Invalid beta_schedule '{self.beta_schedule}'. Choose 'linear', 'quadratic', or 'cosine'.")
        betas = torch.clamp(betas, min=self.eps, max=self.max_beta)
        return betas
    
    def _compute_diffusion_constants(self):
        """Compute all diffusion constants from beta schedule."""
        with torch.no_grad():
            # Compute alpha and alpha_hat
            self.alpha = 1.0 - self.beta
            self.alpha_hat = torch.cumprod(torch.clamp(self.alpha, min=self.eps), dim=0)
            #self.alpha_hat = torch.cumprod(self.alpha, dim=0)
            
            # Precompute sqrt terms for efficiency
            self.sqrt_alpha = torch.sqrt(torch.clamp(self.alpha, min=self.eps))
            self.sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - self.alpha_hat, min=self.eps))
            
            # Compute posterior variance/std (for sampling)
            alpha_hat_tm1 = torch.cat([
                torch.ones(1, device=self.device, dtype=self.dtype),
                self.alpha_hat[:-1]
            ], dim=0)
            posterior_var = self.beta * (1.0 - alpha_hat_tm1) / torch.clamp(1.0 - self.alpha_hat, min=self.eps)
            self.posterior_var = torch.clamp(posterior_var, min=self.eps)
            self.posterior_std = torch.sqrt(self.posterior_var)
    
    def register_to_model(self, model):
        """Register noise schedule and precomputed constants to the model.
        
        Args:
            model: The model to register the noise schedule to
            
        Returns:
            The model with registered noise schedule
        """
        # Use the model's set_noise_schedule method if available
        if hasattr(model, 'set_noise_schedule'):
            model.set_noise_schedule(self.beta, eps=self.eps)
        else:
            # Fallback to direct buffer registration
            model.register_buffer('beta', self.beta)
            model.register_buffer('alpha', self.alpha)
            model.register_buffer('alpha_hat', self.alpha_hat)
            model.register_buffer('sqrt_alpha', self.sqrt_alpha)
            model.register_buffer('sqrt_one_minus_alpha_bar', self.sqrt_one_minus_alpha_bar)
            model.register_buffer('posterior_std', self.posterior_std)
            
            # Set number of timesteps in the model
            model.num_timesteps = self.num_timesteps
        
        return model



def test_noise_schedulers(num_timesteps=1000, plot=True):
    """Test all noise schedulers and optionally plot the results.
    
    Args:
        num_timesteps: Number of diffusion timesteps to test
        plot: Whether to plot the results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create schedulers
    linear_scheduler = NoiseScheduler(num_timesteps=num_timesteps, beta_schedule="linear", device=device)
    quadratic_scheduler = NoiseScheduler(num_timesteps=num_timesteps, beta_schedule="quadratic", device=device)
    cosine_scheduler = NoiseScheduler(num_timesteps=num_timesteps, beta_schedule="cosine", device=device)
    
    # Get betas
    linear_betas = linear_scheduler.beta.cpu().numpy()
    quadratic_betas = quadratic_scheduler.beta.cpu().numpy()
    cosine_betas = cosine_scheduler.beta.cpu().numpy()
    
    # Get alphas
    linear_alphas = linear_scheduler.alpha_hat.cpu().numpy()
    quadratic_alphas = quadratic_scheduler.alpha_hat.cpu().numpy()
    cosine_alphas = cosine_scheduler.alpha_hat.cpu().numpy()
    
    # Print statistics
    print("Linear schedule:")
    print(f"  Beta range: {linear_betas.min():.6f} to {linear_betas.max():.6f}")
    print(f"  Alpha_hat range: {linear_alphas.min():.6f} to {linear_alphas.max():.6f}")
    
    print("\nQuadratic schedule:")
    print(f"  Beta range: {quadratic_betas.min():.6f} to {quadratic_betas.max():.6f}")
    print(f"  Alpha_hat range: {quadratic_alphas.min():.6f} to {quadratic_alphas.max():.6f}")
    
    print("\nCosine schedule:")
    print(f"  Beta range: {cosine_betas.min():.6f} to {cosine_betas.max():.6f}")
    print(f"  Alpha_hat range: {cosine_alphas.min():.6f} to {cosine_alphas.max():.6f}")
    
    if plot:
        # Plot betas
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(linear_betas, label='Linear')
        plt.plot(quadratic_betas, label='Quadratic')
        plt.plot(cosine_betas, label='Cosine')
        plt.title('Beta Schedules')
        plt.xlabel('Timestep')
        plt.ylabel('Beta')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(linear_alphas, label='Linear')
        plt.plot(quadratic_alphas, label='Quadratic')
        plt.plot(cosine_alphas, label='Cosine')
        plt.title('Alpha_hat Schedules')
        plt.xlabel('Timestep')
        plt.ylabel('Alpha_hat')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('noise_schedules.pdf')
        plt.show()
    
    return {
        'linear': {'betas': linear_betas, 'alphas': linear_alphas},
        'quadratic': {'betas': quadratic_betas, 'alphas': quadratic_alphas},
        'cosine': {'betas': cosine_betas, 'alphas': cosine_alphas}
    }


if __name__ == "__main__":
    print("Testing noise schedulers...")
    
    # Test with different timesteps
    for timesteps in [100, 500, 1000]:
        print(f"\n=== Testing with {timesteps} timesteps ===")
        results = test_noise_schedulers(num_timesteps=timesteps, plot=(timesteps == 1000))
        
        # Additional verification
        for schedule_name, data in results.items():
            betas, alphas = data['betas'], data['alphas']
            
            # Check for monotonicity in alpha_hat (should be decreasing)
            is_monotonic = all(alphas[i] >= alphas[i+1] for i in range(len(alphas)-1))
            print(f"{schedule_name.capitalize()} alpha_hat is monotonically decreasing: {is_monotonic}")
    
    print("\nTesting complete! Check the generated plot for visualization.")
