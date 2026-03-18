#!/usr/bin/env python3
"""
Script to pre-generate Hodgkin-Huxley data and save it for later use.
This can be used as a fallback when JAX/Symformer has issues.
"""

import os
import sys
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate Hodgkin-Huxley data')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--output_path', type=str, default='/cephyr/users/nautiyal/Alvis/diffusion/hh_task_data.npz', 
                        help='Path to save the generated data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--observation_seed', type=int, default=18, help='Seed for observation generation')
    args = parser.parse_args()
    
    print(f"Setting environment variables for JAX...")
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    try:
        import jax
        print(f"JAX version: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")
        
        # Try to force JAX to use GPU
        jax.config.update('jax_platform_name', 'gpu')
        
        from jax.lib import xla_bridge
        backend = xla_bridge.get_backend().platform
        print(f"JAX backend: {backend}")
        
        # Import HH task
        try:
            from scoresbibm.tasks.hhtask import HHTask
            print("Successfully imported HHTask")
        except ImportError:
            print("Error importing HHTask. Installing scoresbibm...")
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scoresbibm'])
            from scoresbibm.tasks.hhtask import HHTask
            print("Successfully installed and imported HHTask")
        
        # Create HH task
        print("Creating HH task...")
        hh_task = HHTask(backend="jax")
        
        # Generate data
        print(f"Generating {args.num_samples} samples...")
        rng = jax.random.PRNGKey(args.seed)
        data = hh_task.get_data(num_samples=args.num_samples, rng=rng)
        
        # Convert to numpy arrays
        theta = np.asarray(data["theta"])
        x = np.asarray(data["x"])
        print(f"Generated data shapes: theta={theta.shape}, x={x.shape}")
        
        # Generate observation
        print("Generating observation...")
        key = jax.random.PRNGKey(args.observation_seed)
        key, obs_key = jax.random.split(key)
        
        observation_generator = hh_task.get_observation_generator(condition_mask_fn="posterior")
        condition_mask, x_o, theta_o = next(observation_generator(obs_key))
        
        key, sim_key = jax.random.split(key)
        simulator = hh_task.get_simulator()
        V, H, _ = simulator(sim_key, theta_o)
        
        observation = {
            "theta_o": np.asarray(theta_o),
            "observed_data": np.asarray(x_o),
            "voltage_trace": np.asarray(V),
            "energy_trace": np.asarray(H),
            "condition_mask": np.asarray(condition_mask),
        }
        
        # Save data
        print(f"Saving data to {args.output_path}...")
        np.savez(args.output_path, theta=theta, x=x, observation=observation)
        print("Data saved successfully!")
        
    except Exception as e:
        print(f"Error generating HH data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
