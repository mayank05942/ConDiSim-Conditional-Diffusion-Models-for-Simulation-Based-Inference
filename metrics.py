import os
import json
import numpy as np
import torch
import sbibm
import time
import hashlib
from sbibm.metrics import c2st
from torch.nn.functional import pairwise_distance

METRIC_TYPE = 'c2st' 
results_dir = "results"
task_names = ['sir']

def compute_mmd(P, Q, kernel='inverse_multiquadratic', scales=None):
    if scales is None:
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
        dist = torch.sum((x[:, None, :] - y[None, :, :]) ** 2, dim=-1, keepdim=True)
        sigmas = scales.view(1, 1, -1)
        return torch.sum(sigmas / (dist + sigmas), dim=-1)
    
    K_PP = inverse_multiquadratic(P, P, scales)
    K_QQ = inverse_multiquadratic(Q, Q, scales)
    K_PQ = inverse_multiquadratic(P, Q, scales)
    
    mmd = K_PP.mean() + K_QQ.mean() - 2 * K_PQ.mean()
    return mmd.item()

if METRIC_TYPE == 'c2st':
    metric_results_dir = "c2st_results"
    metric_key = 'c2st_accuracy'
elif METRIC_TYPE == 'mmd':
    metric_results_dir = "mmd_results"
    metric_key = 'mmd_score'
else:
    raise ValueError(f"Unknown METRIC_TYPE: {METRIC_TYPE}. Must be 'c2st' or 'mmd'")

print(f"\n{'='*60}")
print(f"METRIC TYPE: {METRIC_TYPE.upper()}")
print(f"Results will be saved to: {metric_results_dir}")
print(f"{'='*60}\n")

checkpoint_file = os.path.join(metric_results_dir, "progress_checkpoint.json")

processed_files = {}
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        processed_files = json.load(f)
    print(f"Loaded checkpoint with {len(processed_files)} processed files")

for task_name in task_names:
    task_path = os.path.join(results_dir, task_name)
    if os.path.isdir(task_path):
        print(f"Processing task: {task_name}")
        
        task_results_dir = os.path.join(metric_results_dir, task_name)
        if not os.path.exists(task_results_dir):
            os.makedirs(task_results_dir)

        result_file = os.path.join(task_results_dir, f'{task_name}_{METRIC_TYPE}_results.json')

        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                metric_results = json.load(f)
        else:
            metric_results = {}

        task = sbibm.get_task(task_name)
        true_posterior = task.get_reference_posterior_samples(num_observation=1)

        true_posterior = torch.tensor(true_posterior.numpy(), dtype=torch.float32)

        npz_files = []
        for file_name in os.listdir(task_path):
            if (file_name.endswith('.npz') and 'run' in file_name and 'budget' in file_name
                and 'intermediate' not in file_name and 'sbc_draws' not in file_name):
                npz_files.append(file_name)
        
        npz_files.sort(key=lambda x: int(x.split('_')[x.split('_').index('budget')+1].replace('.npz', '')))
        
        def check_for_identical_samples():
            print("\nChecking for identical samples across runs...")
            sample_hashes = {}
            identical_found = False
            
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
                    continue
                    
                budget_hashes = {}
                for file_name in files:
                    file_path = os.path.join(task_path, file_name)
                    try:
                        data = np.load(file_path)
                        if 'theta_samples' in data:
                            samples = data['theta_samples'][:10]
                            sample_hash = hashlib.md5(samples.tobytes()).hexdigest()
                            budget_hashes[file_name] = sample_hash
                    except Exception as e:
                        print(f"Error checking {file_name}: {e}")
                
                if len(set(budget_hashes.values())) < len(budget_hashes):
                    identical_found = True
                    print(f"WARNING: Found identical samples across runs for budget {budget}!")
                    print("This will result in identical C2ST scores across runs.")
                    print("Consider using different random seeds for each run in sbi_run.py")
                    
                    files_by_hash = {}
                    for fname, h in budget_hashes.items():
                        if h not in files_by_hash:
                            files_by_hash[h] = []
                        files_by_hash[h].append(fname)
                    
                    for h, file_list in files_by_hash.items():
                        if len(file_list) > 1:
                            print(f"  Identical samples in: {', '.join(file_list)}")
            
            return identical_found
        
        check_for_identical_samples()
        
        print(f"Found {len(npz_files)} files to process for task {task_name}")
        
        for file_index, file_name in enumerate(npz_files):
            file_path = os.path.join(task_path, file_name)
            file_key = f"{task_name}/{file_name}"
            
            if file_key in processed_files:
                print(f"[{file_index+1}/{len(npz_files)}] Skipping {file_name} (already processed according to checkpoint)")
                continue
                
            print(f"[{file_index+1}/{len(npz_files)}] Processing file: {file_path}")

            parts = file_name.split('_')
            if 'budget' in parts and 'run' in parts:
                budget_index = parts.index('budget')
                run_index = parts.index('run')

                simulation_budget = int(parts[budget_index + 1].replace('.npz', ''))
                run_number = int(parts[run_index + 1].replace('.npz', ''))

                if str(simulation_budget) in metric_results and str(run_number) in metric_results[str(simulation_budget)]:
                    print(f"Result for {file_name} already exists in results file, skipping...")
                    processed_files[file_key] = True
                    with open(checkpoint_file, 'w') as f:
                        json.dump(processed_files, f)
                    continue

                data = np.load(file_path)
                theta_samples = data['theta_samples']

                theta_samples = torch.tensor(theta_samples, dtype=torch.float32)

                print(f"Computing {METRIC_TYPE.upper()} metric for {file_name}...")
                start_time = time.time()
                
                if METRIC_TYPE == 'c2st':
                    metric_value = c2st(true_posterior, theta_samples).item()
                elif METRIC_TYPE == 'mmd':
                    metric_value = compute_mmd(true_posterior, theta_samples)
                
                elapsed = time.time() - start_time
                print(f"{METRIC_TYPE.upper()} computation took {elapsed:.2f} seconds")

                if str(simulation_budget) not in metric_results:
                    metric_results[str(simulation_budget)] = {}
                metric_results[str(simulation_budget)][str(run_number)] = {
                    metric_key: metric_value
                }
                
                processed_files[file_key] = True
                
                with open(result_file, 'w') as f:
                    json.dump(metric_results, f, indent=4)
                with open(checkpoint_file, 'w') as f:
                    json.dump(processed_files, f)
                    
                print(f"Added {METRIC_TYPE.upper()} result for budget {simulation_budget}, run {run_number} and saved progress")

        with open(result_file, 'w') as f:
            json.dump(metric_results, f, indent=4)

        print(f"{METRIC_TYPE.upper()} results for task '{task_name}' saved to {result_file}")
