import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import jax
import jax.numpy as jnp
from scoresbibm.tasks.hhtask import HHTask
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import copy
sys.path.append('/cephyr/users/nautiyal/Alvis/diffusion')
from utils import load_data, create_dataloaders, noise_schedule, betas_for_alpha_bar

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearConditionalDiffusionModel(nn.Module):
    def __init__(self, theta_dim, y_dim, layer_sizes, num_timesteps):
        super(LinearConditionalDiffusionModel, self).__init__()
        self.theta_dim = theta_dim
        self.y_dim = y_dim
        self.num_timesteps = num_timesteps

        layers = []
        input_dim = theta_dim + y_dim + 1  # +1 for the time step t
        for size in layer_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(nn.LeakyReLU(0.1))
            input_dim = size
        layers.append(nn.Linear(layer_sizes[-1], theta_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, theta, y, t):
        t_expanded = t.unsqueeze(-1).float() / self.num_timesteps
        input_data = torch.cat([theta, y, t_expanded], dim=-1)
        return self.net(input_data)

    def sample(self, N, y_observed):
        with torch.no_grad():
            intermediate_samples = []
            theta_samples = torch.randn((N, self.theta_dim), device=device)
            for t in reversed(range(self.num_timesteps)):
                if t % (self.num_timesteps // 10) == 0:
                    intermediate_samples.append(theta_samples.cpu().numpy())
                t_tensor = torch.full((N,), t, device=device)
                beta_t = self.beta[t].unsqueeze(-1).to(device)
                alpha_t = self.alpha[t].unsqueeze(-1).to(device)
                alpha_hat_t = self.alpha_hat[t].unsqueeze(-1).to(device)

                theta_pred = self.forward(theta_samples, y_observed, t_tensor)
                mean = (theta_samples - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t + 1e-5)) * theta_pred) / torch.sqrt(alpha_t + 1e-5)

                if t > 0:
                    noise = torch.randn_like(theta_samples)
                    theta_samples = mean + torch.sqrt(beta_t + 1e-5) * noise
                else:
                    theta_samples = mean

            intermediate_samples.append(theta_samples.cpu().numpy())
            return theta_samples, intermediate_samples

def diffusion_loss(model, theta_0, y, num_timesteps, mse_loss):
    t = torch.randint(0, num_timesteps, (theta_0.size(0),), device=device)
    noise = torch.randn_like(theta_0, device=device)
    alpha_hat_t = model.alpha_hat[t].unsqueeze(-1)
    theta_noisy = torch.sqrt(alpha_hat_t + 1e-5) * theta_0 + torch.sqrt(1 - alpha_hat_t + 1e-5) * noise
    theta_pred = model(theta_noisy, y, t)
    return mse_loss(theta_pred, noise)

def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, num_timesteps):
    mse_loss = nn.MSELoss()
    best_val_loss = float('inf')
    best_model = None
    counter = 0
    patience = 20  # Early stopping patience

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for theta_batch, y_batch in train_loader:
            theta_batch = theta_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            loss = diffusion_loss(model, theta_batch, y_batch, num_timesteps, mse_loss)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for theta_batch, y_batch in val_loader:
                theta_batch = theta_batch.to(device)
                y_batch = y_batch.to(device)
                val_loss += diffusion_loss(model, theta_batch, y_batch, num_timesteps, mse_loss).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}, Learning Rate: {current_lr:.6f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    return best_model, train_losses, val_losses

def create_hh_dataset(simulation_budget, seed=42):
    """
    Create a dataset of Hodgkin-Huxley model simulations.
    
    Args:
        simulation_budget (int): Number of simulations to run
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (theta, x) where:
            - theta: Parameters used for simulations (shape: [simulation_budget, num_params])
            - x: Corresponding voltage traces (shape: [simulation_budget, time_points])
    """
    hh_task = HHTask(backend="jax")
    rng = jax.random.PRNGKey(seed)
    
    # Generate data
    data = hh_task.get_data(num_samples=simulation_budget, rng=rng)
    
    # Extract parameters (theta) and corresponding simulated data (x)
    theta = data["theta"]
    x = data["x"]
    
    print(f"Generated dataset with {simulation_budget} samples:")
    print(f"Parameters shape: {theta.shape}")
    return theta, x

def generate_observed_data(hh_task, seed=18):
    """
    Generate observed data including voltage trace, energy, and summary statistics.
    
    Args:
        hh_task: HHTask instance
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (V, H, summary_stats, summary_stats_with_energy) where:
            - V: Voltage trace
            - H: Energy trace
            - summary_stats: Original summary statistics
            - summary_stats_with_energy: Summary stats including total energy
    """
    simulator = hh_task.get_simulator()
    observation_generator = hh_task.get_observation_generator(condition_mask_fn="posterior")
    key = jax.random.PRNGKey(seed)

    key, subkey = jax.random.split(key)
    observation_stream = observation_generator(key)
    condition_mask, x_o, theta_o = next(observation_stream)
    
    # Generate traces and stats
    V, H, _ = simulator(key, theta_o)
    
    print(f"Generated observed data:")
    print(f"Parameters: {theta_o.shape}")
    print(f"Observation shape: {x_o.shape}")
    print(f"Voltage trace shape: {V.shape}")
    print(f"Energy trace shape: {H.shape}")
    
    return theta_o, x_o, V, H

if __name__ == "__main__":
    device = torch.device("cpu")
    # Define hyperparameters
    hyperparams = {
        'theta_dim': 7,
        'y_dim': 8,
        'num_timesteps': 300,
        'learning_rate': 1e-3,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'patience': 20,
        'batch_size': 32,
        'num_epochs': 1000,
        'beta_schedule': 'quadratic',
        'layer_sizes': [256,256,236],
        'simulation_budgets': [10000],
        'num_runs': 1,
        'num_samples': 10000  
    }

    # Create results directory if it doesn't exist
    results_dir = "results"  # We're already in the hh directory
    os.makedirs(results_dir, exist_ok=True)

    # Iterate over simulation budgets and runs
    for budget in hyperparams['simulation_budgets']:
        for run in range(1, hyperparams['num_runs'] + 1):
            print(f"\nStarting simulation budget {budget}, run {run}")
            
            # Generate dataset
            hh_task = HHTask(backend="jax")
            theta, x = create_hh_dataset(simulation_budget=budget)

            theta = np.array(theta)
            x = np.array(x)

            # Apply standard scaling
            scaler_theta = StandardScaler()
            scaler_x = StandardScaler()

            theta = scaler_theta.fit_transform(theta)
            x = scaler_x.fit_transform(x)

            # Convert to torch tensors
            theta = torch.tensor(theta, dtype=torch.float32)
            x = torch.tensor(x, dtype=torch.float32)

            # Generate observed data (same for all runs of same budget)
            theta_o, x_o, V, H = generate_observed_data(hh_task)

            # Instantiate the LinearConditionalDiffusionModel
            theta_dim = hyperparams['theta_dim']
            y_dim = hyperparams['y_dim']
            layer_sizes = hyperparams['layer_sizes']
            num_timesteps = hyperparams['num_timesteps']

            diffusion_model = LinearConditionalDiffusionModel(theta_dim, y_dim, layer_sizes, num_timesteps).to(device)

            # Initialize beta schedule and compute alpha, alpha_hat
            beta = noise_schedule(
                num_timesteps=num_timesteps,
                beta_start=hyperparams['beta_start'],
                beta_end=hyperparams['beta_end'],
                beta_schedule=hyperparams['beta_schedule']
            )
            alpha = 1 - beta
            alpha_hat = torch.cumprod(alpha, dim=0)
            
            # Set the parameters in the model
            diffusion_model.beta = beta
            diffusion_model.alpha = alpha
            diffusion_model.alpha_hat = alpha_hat

            # Define optimizer and scheduler
            optimizer = optim.AdamW(diffusion_model.parameters(), lr=hyperparams['learning_rate'], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

            # Prepare data loaders
            train_loader, val_loader = create_dataloaders(theta, x, batch_size=hyperparams['batch_size'])

            # Train the model
            best_model, train_losses, val_losses = train_model(diffusion_model, train_loader, val_loader, optimizer, scheduler, 
                                                             num_epochs=hyperparams['num_epochs'], 
                                                             num_timesteps=num_timesteps)

            # Save the trained model
            model_filename = f"hh_diffusion_model_budget_{budget}_run_{run}.pth"
            model_path = os.path.join(results_dir, model_filename)
            torch.save(best_model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

            # Prepare observed data for sampling
            x_o_scaled = scaler_x.transform(x_o.reshape(1, -1))
            x_o_tensor = torch.tensor(x_o_scaled, dtype=torch.float32)
            
            # Sample from the model
            num_samples = hyperparams['num_samples']
            y_observed = x_o_tensor.repeat(num_samples, 1).to(device)
            theta_samples, intermediate_samples = best_model.sample(num_samples, y_observed)
            
            # Transform samples back to original scale
            theta_samples_descaled = scaler_theta.inverse_transform(theta_samples.cpu().numpy())

            # Save posterior samples with budget and run information
            posterior_filename = f"hh_posterior_samples_budget_{budget}_run_{run}.npz"
            posterior_path = os.path.join(results_dir, posterior_filename)
            np.savez(posterior_path, 
                     theta_samples=theta_samples_descaled,
                     true_parameters=theta_o,
                     observed_data=x_o,
                     true_V=V,
                     true_H=H,
                     budget=budget,
                     run=run)
            print(f"Posterior samples saved to {posterior_path}")
