import os
import json
import numpy as np
import torch
import sbibm
import time
import hashlib
from sbibm.metrics import c2st
from torch.nn.functional import pairwise_distance

# ============================================
# GLOBAL CONFIGURATION
# ============================================
# Set metric type: 'c2st' or 'mmd'
METRIC_TYPE = 'c2st'  

# Function to compute Maximum Mean Discrepancy (MMD)
def compute_mmd(P, Q, kernel='inverse_multiquadratic', scales=None):
    """
    Compute Maximum Mean Discrepancy between two distributions.
    
    Args:
        P: Reference samples (tensor) of shape (num_draws_P, num_features)
        Q: Generated samples (tensor) of shape (num_draws_Q, num_features)
        kernel: Kernel type (default: 'inverse_multiquadratic')
        scales: Tensor of scales for kernel mixture (default: [0.1, 0.5, 1.0, 2.0, 5.0])
    
    Returns:
        MMD score (float)
    """
    if scales is None:
        # Default scales for inverse multiquadratic kernel
        # Hard coded from logspace(-6, 6, 11) to avoid pytorch errors/warnings
        scales = torch.tensor([
            1.0000e-06,
            1.5849e-05,
            2.5119e-04,
            3.9811e-03,
            6.3096e-02,
            1.0000e00,
            1.5849e01,
            2.5119e02,
            3.9811e03,
            6.3096e04,
            1.0000e06,
        ], device=P.device, dtype=P.dtype)
    
    def inverse_multiquadratic(x, y, scales):
        """
        Computes a mixture of inverse multiquadratic RBFs between samples of x and y.
        
        Args:
            x: Tensor of shape (num_draws_x, num_features)
            y: Tensor of shape (num_draws_y, num_features)
            scales: Tensor of scales for the IM-RBF kernel mixture
        
        Returns:
            kernel_matrix: Tensor of shape (num_draws_x, num_draws_y)
        """
        # Compute squared distances: (num_draws_x, num_draws_y)
        dist = torch.sum((x[:, None, :] - y[None, :, :]) ** 2, dim=-1, keepdim=True)
        # Expand scales: (1, 1, num_scales)
        sigmas = scales.view(1, 1, -1)
        # Compute kernel: sum over scales
        return torch.sum(sigmas / (dist + sigmas), dim=-1)
    
    K_PP = inverse_multiquadratic(P, P, scales)
    K_QQ = inverse_multiquadratic(Q, Q, scales)
    K_PQ = inverse_multiquadratic(P, Q, scales)
    
    mmd = K_PP.mean() + K_QQ.mean() - 2 * K_PQ.mean()
    return mmd.item()

# Define the paths
results_dir = "/cephyr/users/nautiyal/Alvis/diffusion/results"

# Set results directory based on metric type
if METRIC_TYPE == 'c2st':
    metric_results_dir = "/cephyr/users/nautiyal/Alvis/diffusion/c2st_results"
    metric_key = 'c2st_accuracy'
elif METRIC_TYPE == 'mmd':
    metric_results_dir = "/cephyr/users/nautiyal/Alvis/diffusion/mmd_results"
    metric_key = 'mmd_score'
else:
    raise ValueError(f"Unknown METRIC_TYPE: {METRIC_TYPE}. Must be 'c2st' or 'mmd'")

print(f"\n{'='*60}")
print(f"METRIC TYPE: {METRIC_TYPE.upper()}")
print(f"Results will be saved to: {metric_results_dir}")
print(f"{'='*60}\n")

# List of tasks to process
#task_names = [ 'two_moons', 'bernoulli_glm', 'bernoulli_glm_raw', 'gaussian_linear', 'gaussian_linear_uniform', 'gaussian_mixture', 'lotka_volterra', 'sir', 'slcp', 'slcp_distractors']

task_names = ['sir']


# Checkpoint file to track overall progress
checkpoint_file = os.path.join(metric_results_dir, "progress_checkpoint.json")

# Load checkpoint if it exists
processed_files = {}
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        processed_files = json.load(f)
    print(f"Loaded checkpoint with {len(processed_files)} processed files")

# Iterate over all specified task names
for task_name in task_names:
    task_path = os.path.join(results_dir, task_name)
    if os.path.isdir(task_path):  # Check if it is a directory
        print(f"Processing task: {task_name}")
        
        # Create the directory for storing metric results for the task if it does not exist
        task_results_dir = os.path.join(metric_results_dir, task_name)
        if not os.path.exists(task_results_dir):
            os.makedirs(task_results_dir)

        # Path to the consolidated JSON file for this task
        result_file = os.path.join(task_results_dir, f'{task_name}_{METRIC_TYPE}_results.json')

        # Load existing results if the JSON file already exists
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                metric_results = json.load(f)
        else:
            metric_results = {}

        # Load the true posterior samples for the current task
        task = sbibm.get_task(task_name)
        true_posterior = task.get_reference_posterior_samples(num_observation=1)

        # Convert true posterior to PyTorch tensor
        true_posterior = torch.tensor(true_posterior.numpy(), dtype=torch.float32)

        # Get list of all .npz files for this task
        npz_files = []
        for file_name in os.listdir(task_path):
            # Only include files that match the desired format and skip 'intermediate' and 'sbc_draws'
            if (file_name.endswith('.npz') and 'run' in file_name and 'budget' in file_name
                and 'intermediate' not in file_name and 'sbc_draws' not in file_name):
                npz_files.append(file_name)
        
        # Sort files by budget (smaller first) to get quicker results
        npz_files.sort(key=lambda x: int(x.split('_')[x.split('_').index('budget')+1].replace('.npz', '')))
        
        # Check for identical samples across runs with the same budget (without affecting processing)
        def check_for_identical_samples():
            print("\nChecking for identical samples across runs...")
            sample_hashes = {}
            identical_found = False
            
            # Group files by budget
            budget_files = {}
            for file_name in npz_files:
                parts = file_name.split('_')
                if 'budget' in parts:
                    budget_index = parts.index('budget')
                    budget = parts[budget_index + 1].replace('.npz', '')
                    if budget not in budget_files:
                        budget_files[budget] = []
                    budget_files[budget].append(file_name)
            
            # Check each budget group
            for budget, files in budget_files.items():
                if len(files) <= 1:
                    continue  # Skip if only one file for this budget
                    
                # Load and hash a small portion of each file
                budget_hashes = {}
                for file_name in files:
                    file_path = os.path.join(task_path, file_name)
                    try:
                        data = np.load(file_path)
                        if 'theta_samples' in data:
                            # Just use first few samples for quick hash
                            samples = data['theta_samples'][:10]
                            sample_hash = hashlib.md5(samples.tobytes()).hexdigest()
                            budget_hashes[file_name] = sample_hash
                    except Exception as e:
                        print(f"Error checking {file_name}: {e}")
                
                # Check for duplicates
                if len(set(budget_hashes.values())) < len(budget_hashes):
                    identical_found = True
                    print(f"\n⚠️ WARNING: Found identical samples across runs for budget {budget}!")
                    print("This will result in identical C2ST scores across runs.")
                    print("Consider using different random seeds for each run in sbi_run.py")
                    
                    # Group files by hash
                    files_by_hash = {}
                    for fname, h in budget_hashes.items():
                        if h not in files_by_hash:
                            files_by_hash[h] = []
                        files_by_hash[h].append(fname)
                    
                    # Print groups of identical files
                    for h, file_list in files_by_hash.items():
                        if len(file_list) > 1:
                            print(f"  Identical samples in: {', '.join(file_list)}")
            
            return identical_found
        
        # Check for identical samples (informational only)
        check_for_identical_samples()
        
        print(f"Found {len(npz_files)} files to process for task {task_name}")
        
        # Iterate over all .npz files for each task
        for file_index, file_name in enumerate(npz_files):
            file_path = os.path.join(task_path, file_name)
            file_key = f"{task_name}/{file_name}"
            
            # Check if file was already processed (from checkpoint)
            if file_key in processed_files:
                print(f"[{file_index+1}/{len(npz_files)}] Skipping {file_name} (already processed according to checkpoint)")
                continue
                
            print(f"[{file_index+1}/{len(npz_files)}] Processing file: {file_path}")

            # Extract simulation budget and run number from the file name
            parts = file_name.split('_')
            if 'budget' in parts and 'run' in parts:
                budget_index = parts.index('budget')
                run_index = parts.index('run')

                # Extract the simulation budget and run number, removing the '.npz' extension
                simulation_budget = int(parts[budget_index + 1].replace('.npz', ''))
                run_number = int(parts[run_index + 1].replace('.npz', ''))

                # Check if this result already exists in the JSON data
                if str(simulation_budget) in metric_results and str(run_number) in metric_results[str(simulation_budget)]:
                    print(f"Result for {file_name} already exists in results file, skipping...")
                    # Mark as processed in checkpoint
                    processed_files[file_key] = True
                    with open(checkpoint_file, 'w') as f:
                        json.dump(processed_files, f)
                    continue

                # Load the generated samples from the file
                data = np.load(file_path)
                theta_samples = data['theta_samples']

                # Convert generated samples to PyTorch tensor
                theta_samples = torch.tensor(theta_samples, dtype=torch.float32)

                # Compute metric based on METRIC_TYPE
                print(f"Computing {METRIC_TYPE.upper()} metric for {file_name}...")
                start_time = time.time()
                
                if METRIC_TYPE == 'c2st':
                    metric_value = c2st(true_posterior, theta_samples).item()
                elif METRIC_TYPE == 'mmd':
                    metric_value = compute_mmd(true_posterior, theta_samples)
                
                elapsed = time.time() - start_time
                print(f"{METRIC_TYPE.upper()} computation took {elapsed:.2f} seconds")

                # Store the result in the nested dictionary
                if str(simulation_budget) not in metric_results:
                    metric_results[str(simulation_budget)] = {}
                metric_results[str(simulation_budget)][str(run_number)] = {
                    metric_key: metric_value
                }
                
                # Mark as processed in checkpoint
                processed_files[file_key] = True
                
                # Save both the results and checkpoint after each file
                # This ensures we don't lose progress if the job is terminated
                with open(result_file, 'w') as f:
                    json.dump(metric_results, f, indent=4)
                with open(checkpoint_file, 'w') as f:
                    json.dump(processed_files, f)
                    
                print(f"Added {METRIC_TYPE.upper()} result for budget {simulation_budget}, run {run_number} and saved progress")

        # Save the updated metric results to the JSON file
        with open(result_file, 'w') as f:
            json.dump(metric_results, f, indent=4)

        print(f"{METRIC_TYPE.upper()} results for task '{task_name}' saved to {result_file}")
