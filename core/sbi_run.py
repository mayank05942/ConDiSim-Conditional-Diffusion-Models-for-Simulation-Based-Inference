#!/usr/bin/env python3
# SBI problem setup, training, and sampling with diffusion model

import os
import sys
import time
import torch
import numpy as np
import argparse
import sbibm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
# Import bayesflow for simulators
import bayesflow as bf

# Import custom modules
# Use relative imports when running from within the core directory
from model_architecture import ReverseDiffusionModel
from noise_scheduler import NoiseScheduler
from model_train import train_model, create_optimizer_and_scheduler, save_model
from sampling import sample_and_save
from train_utils import set_seed, create_dataloaders, print_network_architecture


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Task parameters
    parser.add_argument("--task_name", type=str, default="gaussian_linear")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--simulation_budgets", type=int, nargs='+', default=[10000])
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--run_number", type=int, default=1, help="Current run number (used for file naming)")
    
    
    # Model parameters
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="quadratic", choices=["linear", "quadratic", "cosine"])

    # Model architecture hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--code_dim", type=int, default=None)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--cfg_dropout_prob", type=float, default=0.2)
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_snr_weighting", action="store_true")
    parser.add_argument("--patience", type=int, default=15)
    
    # Sampling parameters
    parser.add_argument("--posterior_samples", type=int, default=10000)
    parser.add_argument("--observation_index", type=int, default=1)
    parser.add_argument("--lambda_guidance", type=float, default=3.0, help="CFG guidance strength parameter (0.0 = no guidance, >0.0 = stronger guidance)")
    
    # Other parameters
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42 to match hh_main.py)")
    parser.add_argument("--observation_seed", type=int, default=18, help="Seed for generating the observation (default: 18 to match hh_main.py)")
    parser.add_argument("--device", type=str, default=None)
    
    return parser.parse_args()


def generate_hh_observation(hh_task, seed: int = 18):
    """Generate the Hodgkin-Huxley observation and auxiliary traces.
    
    This function matches the implementation in hh_main.py's generate_observed_data function.
    """
    try:
        import jax
        print(f"Generating HH observation with seed {seed}")
        
        # Create the simulator and observation generator
        simulator = hh_task.get_simulator()
        observation_generator = hh_task.get_observation_generator(condition_mask_fn="posterior")
        
        # Create and split PRNG key - exactly as in hh_main.py
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        
        # Generate observation - exactly as in hh_main.py
        observation_stream = observation_generator(key)
        condition_mask, x_o, theta_o = next(observation_stream)
        
        # Generate traces - exactly as in hh_main.py
        V, H, _ = simulator(key, theta_o)
        
        print(f"Generated observed data:")
        print(f"Parameters: {theta_o.shape}")
        print(f"Observation shape: {x_o.shape}")
        print(f"Voltage trace shape: {V.shape}")
        print(f"Energy trace shape: {H.shape}")
        
        # Convert to numpy arrays for PyTorch compatibility
        result = {
            "theta_o": np.asarray(theta_o),
            "observed_data": np.asarray(x_o),
            "voltage_trace": np.asarray(V),
            "energy_trace": np.asarray(H),
            "condition_mask": np.asarray(condition_mask),
        }
        
        return result
        
    except Exception as e:
        print(f"Error in generate_hh_observation: {e}")
        import traceback
        traceback.print_exc()
        raise


def prepare_hh_data(num_samples: int, seed: int = 42, observation_seed: int = 18):
    """Create Hodgkin-Huxley training data and attach observation payload.
    
    This function matches the implementation in hh_main.py's create_hh_dataset function.
    
    Args:
        num_samples (int): Number of simulations to run
        seed (int): Random seed for reproducibility
        observation_seed (int): Seed for generating the observation
        
    Returns:
        tuple: (theta, x, hh_task) where:
            - theta: Parameters used for simulations
            - x: Corresponding voltage traces
            - hh_task: HHTask instance with observation payload
    """
    try:
        # Set environment variables to help JAX work better
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        import jax
        print(f"JAX version: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")
        
        # Import HH task
        from scoresbibm.tasks.hhtask import HHTask
        
        # Create HH task with jax backend - exactly as in hh_main.py
        hh_task = HHTask(backend="jax")
        
        # Generate data - exactly as in hh_main.py
        rng = jax.random.PRNGKey(seed)
        print(f"Generating {num_samples} samples...")
        data = hh_task.get_data(num_samples=num_samples, rng=rng)
        
        # Extract parameters and data - exactly as in hh_main.py
        theta = np.asarray(data["theta"])
        x = np.asarray(data["x"])
        
        print(f"Generated dataset with {num_samples} samples:")
        print(f"Parameters shape: {theta.shape}")
        
        # Generate observation data - exactly as in hh_main.py
        print("\nGenerating observation data...")
        theta_o, x_o, V, H = generate_observed_data(hh_task, seed=observation_seed)
        
        # Store observation payload in the task
        hh_task.observation_payload = {
            "theta_o": theta_o,
            "observed_data": x_o,
            "voltage_trace": V,
            "energy_trace": H,
            "condition_mask": None,  # Not used in hh_main.py
        }
        
        return theta, x, hh_task
        
    except Exception as e:
        print(f"Error in prepare_hh_data: {e}")
        import traceback
        traceback.print_exc()
        raise


def generate_observed_data(hh_task, seed=18):
    """Generate observed data including voltage trace and energy.
    
    This is an exact copy of the function from hh_main.py.
    
    Args:
        hh_task: HHTask instance
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (theta_o, x_o, V, H) where:
            - theta_o: True parameters
            - x_o: Observation data
            - V: Voltage trace
            - H: Energy trace
    """
    # Import jax here to ensure it's available in this function
    import jax
    
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


def load_sbi_data(task_name, num_samples, seed=42, observation_seed=18):
    """Load data from an SBI task.
    
    Args:
        task_name: Name of the SBI task
        num_samples: Number of samples to generate
        seed: Random seed for data generation
        observation_seed: Seed for generating the observation
        
    Returns:
        Tuple of (thetas, y_data, theta_scaler, y_scaler, task)
    """
    print(f"Loading {task_name} task with {num_samples} samples...")

    if task_name == "hh":
        try:
            print("\n" + "=" * 80)
            print("Loading Hodgkin-Huxley task data...")
            print("=" * 80)
            
            print(f"Using seed={seed} for data generation and observation_seed={observation_seed} for observation")
            
            thetas, y_data, task = prepare_hh_data(num_samples=num_samples, seed=seed, observation_seed=observation_seed)
            
            print("Successfully loaded HH task data")
            print("=" * 80 + "\n")
        except ImportError as e:
            print(f"\nERROR: Failed to import required modules for HH task: {e}")
            print("This is likely due to missing JAX or Symformer dependencies.")
            print("Make sure the correct modules are loaded and the environment is properly set up.")
            print("Specific recommendations:")
            print("1. Ensure Python 3.10.8 is loaded: module load Python/3.10.8-GCCcore-12.2.0")
            print("2. Ensure CUDA 12.1.1 is loaded: module load CUDA/12.1.1")
            print("3. Ensure JAX is loaded: module load jax/0.4.25-gfbf-2023a-CUDA-12.1.1")
            print("4. Check if Symformer is installed in your environment")
            raise
        except Exception as e:
            print(f"\nERROR: Failed to prepare HH data: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        # Get task from sbibm (used for observations and references)
        task = sbibm.get_task(task_name)

        if task_name == "lotka_volterra":
            # Use bayesflow's direct LotkaVolterra simulator
            print("Using bayesflow LotkaVolterra simulator...")
            lv = bf.simulators.LotkaVolterra()
            
            # Generate synthetic data using sample method
            print(f"Generating {num_samples} synthetic LV samples...")
            samples = lv.sample(num_samples)
            thetas = samples["parameters"]  # Shape: [num_samples, 4]
            y_data = samples["observables"]  # Shape: [num_samples, 20]
            print(f"Generated parameters shape: {thetas.shape}, observations shape: {y_data.shape}")
            
        elif task_name == "sir":
            # Use bayesflow's direct SIR simulator
            print("Using bayesflow SIR simulator...")
            
            sir = bf.simulators.SIR(subsample=10)
            
            # Generate synthetic data using sample method
            print(f"Generating {num_samples} synthetic SIR samples...")
            samples = sir.sample(num_samples)
            thetas = samples["parameters"]
            y_data = samples["observables"]
            print(f"Generated parameters shape: {thetas.shape}, observations shape: {y_data.shape}")
        else:
            # Standard sbibm path for all other tasks
            prior = task.get_prior()
            simulator = task.get_simulator()

            # Generate synthetic data
            print("Generating synthetic data...")
            thetas = prior(num_samples=num_samples)  # Sample parameters from prior
            y_data = simulator(thetas)  # Generate observations using simulator
    
    # Scale the data
    print("Scaling data...")
    theta_scaler = StandardScaler()
    y_scaler = StandardScaler()
    thetas_scaled = theta_scaler.fit_transform(thetas)
    y_data_scaled = y_scaler.fit_transform(y_data)
    
    # Convert to tensors
    # We don't need requires_grad=True here as the diffusion_loss function will handle gradient tracking
    thetas_tensor = torch.tensor(thetas_scaled, dtype=torch.float32)
    y_data_tensor = torch.tensor(y_data_scaled, dtype=torch.float32)
    
    print(f"Data loaded: theta shape={thetas_tensor.shape}, y shape={y_data_tensor.shape}")
    
    return thetas_tensor, y_data_tensor, theta_scaler, y_scaler, task


def setup_model_and_scheduler(theta_dim, y_dim, num_timesteps, beta_schedule, device, hidden_dim, code_dim, num_blocks, cfg_dropout_prob):
    """Set up the diffusion model and noise scheduler.
    
    Args:
        theta_dim: Dimension of parameter space
        y_dim: Dimension of observation space
        num_timesteps: Number of diffusion timesteps
        beta_schedule: Type of noise schedule
        device: Device to use
        hidden_dim: Hidden dimension size for the model
        code_dim: Code/embedding dimension size (if None, defaults to hidden_dim in the model)
        num_blocks: Number of ReverseDiffusionBlocks
        cfg_dropout_prob: Classifier-free guidance dropout probability
        
    Returns:
        Tuple of (model, scheduler)
    """
    print(f"Setting up model with theta_dim={theta_dim}, y_dim={y_dim}")
    print(f"Model hyperparameters: hidden_dim={hidden_dim}, code_dim={code_dim}, num_blocks={num_blocks}, cfg_dropout_prob={cfg_dropout_prob}")

    # Create model
    model = ReverseDiffusionModel(
        theta_dim=theta_dim,
        y_dim=y_dim,
        hidden_dim=hidden_dim,
        code_dim=code_dim,
        num_blocks=num_blocks,
        cfg_dropout_prob=cfg_dropout_prob,
    ).to(device)
    
    # Create noise scheduler
    scheduler = NoiseScheduler(
        num_timesteps=num_timesteps,
        beta_schedule=beta_schedule,
        device=device
    )
    
    # Register noise schedule to model
    scheduler.register_to_model(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
    print(f"Noise schedule: {beta_schedule} with {num_timesteps} timesteps")
    
    # Print detailed network architecture
    print_network_architecture(model)
    
    return model, scheduler


def train_diffusion_model(model, thetas, y_data, args, device):
    """Train the diffusion model.
    
    Args:
        model: The diffusion model
        thetas: Parameter samples
        y_data: Observation data
        args: Command line arguments
        device: Device to use
        
    Returns:
        Tuple of (trained_model, train_losses, val_losses, training_time)
    """
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        thetas, 
        y_data, 
        batch_size=args.batch_size,
        val_split=0.3  # 30% validation split
    )
    
    print("Creating optimizer and scheduler...")
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, 
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    print(f"Starting training for up to {args.num_epochs} epochs...")
    start_time = time.time()
    
    trained_model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        num_timesteps=model.num_timesteps,
        alpha_hat=model.alpha_hat,
        patience=args.patience,
        min_epochs_before_es=20,
        use_snr_weighting=args.use_snr_weighting,
        max_grad_norm=5.0,
        device=device
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/3600:.2f} hours")
    
    return trained_model, train_losses, val_losses, training_time


def plot_training_curves(train_losses, val_losses, task_name, budget, run, results_dir):
    """Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        task_name: Name of the task
        budget: Simulation budget
        run: Run number
        results_dir: Directory to save results
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {task_name}')
    plt.legend()
    plt.grid(True)
    
    # Use absolute path for results directory
    base_dir = "/cephyr/users/nautiyal/Alvis/diffusion"
    
    # If results_dir is not an absolute path, make it absolute
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(base_dir, results_dir)
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(results_dir, task_name, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save plot
    plot_path = os.path.join(plots_dir, f"{task_name}_loss_budget_{budget}_run_{run}.pdf")
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")


def main():
    """Main function to run the SBI pipeline."""
    # Import the patch to fix torch.load (sys is imported at module level)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import patch_sbibm
    
    # Check for JAX and PyTorch compatibility
    try:
        import jax
        print(f"JAX version: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")
        
        # Check if JAX can see GPU
        from jax.lib import xla_bridge
        print(f"JAX backend: {xla_bridge.get_backend().platform}")
        
        # Check PyTorch (torch is imported at module level)
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
            print(f"PyTorch current device: {torch.cuda.current_device()}")
            print(f"PyTorch device name: {torch.cuda.get_device_name(0)}")
        
        # Check for potential conflicts
        print("\nChecking for potential JAX/PyTorch conflicts...")
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        
        # Set environment variables to help JAX and PyTorch coexist
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        print("Set XLA_PYTHON_CLIENT_PREALLOCATE=false to prevent JAX from pre-allocating all GPU memory")
        
        print("JAX and PyTorch compatibility check completed")
    except ImportError as e:
        print(f"Warning: Could not import JAX: {e}")
    except Exception as e:
        print(f"Warning during JAX/PyTorch compatibility check: {e}")
        import traceback
        traceback.print_exc()
    
    # Parse arguments
    args = parse_args()
    
    # Force CUDA to be available if requested
    if args.device == "cuda":
        # Set environment variables to ensure CUDA is detected
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            print(f"CUDA_VISIBLE_DEVICES is set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
        
        # Try to force CUDA detection
        try:
            # Force PyTorch to recognize CUDA
            if not torch.cuda.is_available():
                print("WARNING: CUDA not detected by PyTorch. Attempting to force CUDA detection...")
                # Force device to cuda regardless of torch.cuda.is_available()
                args.device = "cuda:0"
                # Try to create a tensor on GPU to verify it works
                test_tensor = torch.zeros(1, device="cuda:0")
                del test_tensor
                print("Successfully forced CUDA detection!")
            else:
                print("CUDA is available and detected by PyTorch")
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
        except Exception as e:
            print(f"WARNING: CUDA initialization failed: {e}")
            print("Falling back to CPU.")
            args.device = "cpu"
    
    args.device = torch.device(args.device)
    print(f"Using device: {args.device}")
    
    # Don't set a fixed seed - this will use a different random seed each time
    # which ensures different samples for each run
    print("Using system-generated random seed for this run")
    # This approach doesn't affect file naming, only the randomness of the samples
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Print total number of runs planned
    total_runs = len(args.simulation_budgets)
    print(f"\nPlanning to execute {total_runs} total runs (1 run for each of {len(args.simulation_budgets)} budgets)")
    
    # Iterate over simulation budgets and runs
    run_count = 0
    for budget in args.simulation_budgets:
        # Use the provided run_number instead of iterating
        run = args.run_number
        run_count += 1
        print(f"\n{'='*80}")
        print(f"Starting simulation budget {budget}, run {run} (overall progress: {run_count}/{total_runs})")
        print(f"{'='*80}\n")
        
        try:
            # Load data for the specified task
            thetas, y_data, theta_scaler, y_scaler, task = load_sbi_data(
                args.task_name, 
                num_samples=budget,
                seed=args.seed,
                observation_seed=args.observation_seed
            )
            
            # Get dimensions
            theta_dim = thetas.shape[1]
            
            # For HH task, get y_dim from the data
            # For other tasks, it might be defined differently
            if args.task_name == "hh":
                y_dim = y_data.shape[1]
                print(f"HH task dimensions: theta_dim={theta_dim}, y_dim={y_dim}")
            else:
                # For other tasks, use the original approach
                # This assumes y_dim is defined elsewhere for other tasks
                try:
                    # Check if y_dim is already defined
                    y_dim
                except NameError:
                    # If not defined, try to infer it from the data
                    if hasattr(y_data, 'shape') and len(y_data.shape) > 1:
                        y_dim = y_data.shape[1]
                    else:
                        # Default fallback
                        print("Warning: Could not determine y_dim, using default value of 8")
                        y_dim = 8
                print(f"Task dimensions: theta_dim={theta_dim}, y_dim={y_dim}")
            
            # Setup model and scheduler
            model, scheduler = setup_model_and_scheduler(
                theta_dim=theta_dim,
                y_dim=y_dim,
                num_timesteps=args.num_timesteps,
                beta_schedule=args.beta_schedule,
                device=args.device,
                hidden_dim=args.hidden_dim,
                code_dim=args.code_dim,
                num_blocks=args.num_blocks,
                cfg_dropout_prob=args.cfg_dropout_prob,
            )
            
            # Train model
            trained_model, train_losses, val_losses, training_time = train_diffusion_model(
                model=model,
                thetas=thetas,
                y_data=y_data,
                args=args,
                device=args.device
            )
            
            # Save model
            model_path = save_model(
                model=trained_model,
                train_losses=train_losses,
                val_losses=val_losses,
                training_time=training_time,
                task_name=args.task_name,
                budget=budget,
                run=run,
                results_dir=args.results_dir
            )
            
            # Plot training curves
            plot_training_curves(
                train_losses=train_losses,
                val_losses=val_losses,
                task_name=args.task_name,
                budget=budget,
                run=run,
                results_dir=args.results_dir
            )
            
            # Get observation for posterior sampling
            print(f"Getting observation {args.observation_index} for posterior sampling...")
            
            extra_arrays = None
            if args.task_name == "sir":
                # Special case for SIR - load from saved data file
                print("Using special SIR observation data...")
                try:
                    sir_data_path = '/cephyr/users/nautiyal/Alvis/diffusion/sir_task_data.npz'
                    print(f"Loading SIR data from {sir_data_path}")
                    loaded_data = np.load(sir_data_path, allow_pickle=True)
                    true_param = loaded_data['set_1'].item()['true_param']
                    observed_data = loaded_data['set_1'].item()['observed_data'][0]
                    observed_data = observed_data.reshape(1, -1)
                    x_o_scaled = torch.tensor(y_scaler.transform(observed_data), dtype=torch.float32)
                    print(f"SIR observation loaded, shape: {x_o_scaled.shape}")
                except Exception as e:
                    print(f"Error loading SIR data: {e}, falling back to sbibm observation")
                    x_o = task.get_observation(num_observation=args.observation_index)
                    x_o_scaled = torch.tensor(y_scaler.transform(x_o.reshape(1, -1)), dtype=torch.float32)
            elif args.task_name == "hh":
                payload = getattr(task, "observation_payload", None)
                if payload is None:
                    print("HH observation payload missing; regenerating...")
                    payload = generate_hh_observation(task)
                    task.observation_payload = payload

                observed_data = payload["observed_data"].reshape(1, -1)
                x_o_scaled = torch.tensor(y_scaler.transform(observed_data), dtype=torch.float32)

                extra_arrays = {
                    "true_parameters": payload.get("theta_o"),
                    "observed_data": payload.get("observed_data"),
                    "voltage_trace": payload.get("voltage_trace"),
                    "energy_trace": payload.get("energy_trace"),
                    "condition_mask": payload.get("condition_mask"),
                }
            else:
                # Standard case for other tasks
                x_o = task.get_observation(num_observation=args.observation_index)
                x_o_scaled = torch.tensor(y_scaler.transform(x_o.reshape(1, -1)), dtype=torch.float32)

            # Sample from posterior
            print(f"Sampling {args.posterior_samples} samples from posterior...")
            samples, sampling_time = sample_and_save(
                model=trained_model,
                y_observed=x_o_scaled,
                theta_scaler=theta_scaler,
                task_name=args.task_name,
                budget=budget,
                run=run,
                num_samples=args.posterior_samples,
                device=args.device,
                results_dir=args.results_dir,
                lambda_guidance=args.lambda_guidance,
                extra_arrays=extra_arrays,
            )
            
            print(f"\nCompleted budget {budget}, run {run}")
            print(f"Model saved to: {model_path}")
            print(f"Training time: {training_time/60:.2f} minutes")
            print(f"Sampling time: {sampling_time:.2f} seconds")
            
            # Clean up to prevent memory leaks
            del model, trained_model, thetas, y_data, theta_scaler, y_scaler, task
            del train_losses, val_losses, samples
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"CUDA memory cleared after run {run}")
            
        except Exception as e:
            print(f"\nERROR in budget {budget}, run {run}: {str(e)}")
            import traceback
            traceback.print_exc()
            print("\nContinuing with next run...")
    
    print(f"\nAll runs completed successfully! ({run_count}/{total_runs} runs)")
    print(f"Results saved in {os.path.abspath(args.results_dir)}/{args.task_name}/")
    
    # Print a summary of the files created
    results_path = os.path.join(args.results_dir, args.task_name)
    if os.path.exists(results_path):
        print("\nFiles created:")
        for file in sorted(os.listdir(results_path)):
            if file.endswith(".pth") or file.endswith(".npz"):
                print(f"  - {file}")
    
    return run_count  # Return the number of completed runs


if __name__ == "__main__":
    main()
