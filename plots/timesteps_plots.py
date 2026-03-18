# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import sbibm
# import json
# import torch
# import pandas as pd
# import math
# from collections import defaultdict
# from sbibm.metrics.c2st import c2st
# from torch.nn.functional import pairwise_distance

# # Function to compute Jensen-Shannon Divergence (JSD) using PyTorch
# def compute_jsd(P, Q):
#     epsilon = 1e-10
#     P = torch.clamp(P, min=epsilon)
#     Q = torch.clamp(Q, min=epsilon)

#     # Normalize the distributions
#     P = P / (P.sum(dim=1, keepdim=True) + epsilon)
#     Q = Q / (Q.sum(dim=1, keepdim=True) + epsilon)

#     # Calculate the average distribution
#     M = 0.5 * (P + Q)

#     # Compute the KL divergence and then JSD
#     kl_P_M = torch.sum(P * (torch.log(P) - torch.log(M)), dim=1)
#     kl_Q_M = torch.sum(Q * (torch.log(Q) - torch.log(M)), dim=1)
#     jsd_values = 0.5 * (kl_P_M + kl_Q_M)

#     # Return the mean JSD
#     return torch.mean(jsd_values).item()

# # Function to compute C2ST accuracy
# def compute_c2st(reference_samples, generated_samples):
#     c2st_accuracy = c2st(reference_samples, generated_samples)
#     return c2st_accuracy

# # Function to compute Wasserstein Distance
# from scipy.stats import wasserstein_distance

# def compute_wasserstein_distance(P, Q):
#     return wasserstein_distance(P.flatten().numpy(), Q.flatten().numpy())

# # Function to compute KL Divergence
# def compute_kl_divergence(P, Q):
#     epsilon = 1e-10
#     P = torch.clamp(P, min=epsilon)
#     Q = torch.clamp(Q, min=epsilon)

#     # Normalize the distributions
#     P = P / (P.sum(dim=1, keepdim=True) + epsilon)
#     Q = Q / (Q.sum(dim=1, keepdim=True) + epsilon)

#     # Compute KL divergence
#     kl_div = torch.sum(P * (torch.log(P) - torch.log(Q)), dim=1)
#     return torch.mean(kl_div).item()

# # Function to compute Total Variation Distance (TVD)
# def compute_total_variation_distance(P, Q):
#     epsilon = 1e-10
#     P = torch.clamp(P, min=epsilon)
#     Q = torch.clamp(Q, min=epsilon)

#     # Normalize the distributions
#     P = P / (P.sum(dim=1, keepdim=True) + epsilon)
#     Q = Q / (Q.sum(dim=1, keepdim=True) + epsilon)

#     # Compute TVD
#     tvd = 0.5 * torch.sum(torch.abs(P - Q), dim=1)
#     return torch.mean(tvd).item()

# # Function to compute Hellinger Distance
# def compute_hellinger_distance(P, Q):
#     epsilon = 1e-10
#     P = torch.clamp(P, min=epsilon)
#     Q = torch.clamp(Q, min=epsilon)

#     # Normalize the distributions
#     P = P / (P.sum(dim=1, keepdim=True) + epsilon)
#     Q = Q / (Q.sum(dim=1, keepdim=True) + epsilon)

#     # Compute Hellinger distance
#     hellinger_dist = torch.sqrt(torch.sum((torch.sqrt(P) - torch.sqrt(Q))**2, dim=1)) / math.sqrt(2)
#     return torch.mean(hellinger_dist).item()

# # Function to compute Maximum Mean Discrepancy (MMD)
# def compute_mmd(P, Q, kernel='rbf', bandwidth=1.0):
#     def rbf_kernel(x, y, gamma=1.0):
#         dist = pairwise_distance(x, y, p=2)
#         return torch.exp(-gamma * (dist ** 2))

#     K_PP = rbf_kernel(P, P, gamma=1.0 / (2 * (bandwidth ** 2)))
#     K_QQ = rbf_kernel(Q, Q, gamma=1.0 / (2 * (bandwidth ** 2)))
#     K_PQ = rbf_kernel(P, Q, gamma=1.0 / (2 * (bandwidth ** 2)))

#     mmd = K_PP.mean() + K_QQ.mean() - 2 * K_PQ.mean()
#     return mmd.item()

# # Directory where results are stored
# results_dir = '/cephyr/users/nautiyal/Alvis/diffusion/timestep_results'

# # File to save/load evaluation results
# output_file = 'timesteps_results.json'

# # Choose metric: 'JSD', 'C2ST', 'Wasserstein', 'KL', 'TVD', 'Hellinger', 'MMD'
# metric = 'JSD'  # Change this to the desired metric

# # Check if evaluation data exists
# if os.path.exists(output_file):
#     # Load evaluation data from file
#     with open(output_file, 'r') as f:
#         evaluation_results = json.load(f)
#     print(f"Evaluation results loaded from {output_file}")
# else:
#     # Get the task and true posterior samples
#     task_name = 'two_moons'
#     task = sbibm.get_task(task_name)
#     true_posterior = task.get_reference_posterior_samples(num_observation=1).numpy()
#     print("True posterior shape:", true_posterior.shape)

#     # Convert true posterior to PyTorch tensor and clip to avoid issues
#     epsilon = 1e-10
#     true_posterior = torch.tensor(true_posterior, dtype=torch.float32)
#     true_posterior = torch.clamp(true_posterior, min=epsilon)

#     # Initialize dictionary to store evaluation results
#     evaluation_results = defaultdict(list)

#     # Iterate through all subdirectories for different schedulers, T values, and runs
#     for scheduler_type in ['linear', 'quadratic', 'cosine']:
#         for num_timesteps in range(10, 1001, 10):
#             if metric == 'C2ST' and num_timesteps < 30:
#                 # Skip calculations for T < 30 when using C2ST
#                 continue

#             metric_runs = []
#             for run in range(1, 6):  # Assuming 5 runs
#                 # Path to the saved samples for the current T, scheduler, and run
#                 sample_path = os.path.join(results_dir, f"samples_T{num_timesteps}_{scheduler_type}_run{run}.npz")
#                 print(f"Checking file: {sample_path}")

#                 if os.path.exists(sample_path):
#                     print(f"File found: {sample_path}")
#                     # Load generated samples
#                     data = np.load(sample_path)
#                     generated_samples = data['theta_samples']

#                     # Convert generated samples to PyTorch tensor
#                     generated_samples = torch.tensor(generated_samples, dtype=torch.float32)

#                     # Compute the chosen metric between generated and true posterior samples
#                     if metric == 'JSD':
#                         metric_value = compute_jsd(generated_samples, true_posterior)
#                     elif metric == 'C2ST':
#                         metric_value = compute_c2st(true_posterior, generated_samples)
#                     elif metric == 'Wasserstein':
#                         metric_value = compute_wasserstein_distance(generated_samples, true_posterior)
#                     elif metric == 'KL':
#                         metric_value = compute_kl_divergence(generated_samples, true_posterior)
#                     elif metric == 'TVD':
#                         metric_value = compute_total_variation_distance(generated_samples, true_posterior)
#                     elif metric == 'Hellinger':
#                         metric_value = compute_hellinger_distance(generated_samples, true_posterior)
#                     elif metric == 'MMD':
#                         metric_value = compute_mmd(generated_samples, true_posterior)

#                     metric_runs.append(metric_value)
#                 else:
#                     print(f"File not found: {sample_path}")

#             # If there are valid runs, calculate the mean of the metric
#             if metric_runs:
#                 mean_metric = np.mean(metric_runs)
#                 evaluation_results[scheduler_type].append((num_timesteps, mean_metric))

#     # Save results to a file
#     with open(output_file, 'w') as f:
#         json.dump(evaluation_results, f)
#     print(f"Results saved to {output_file}")

# # Define color-blind friendly colors
# color_blind_palette = {
#     'linear': '#E69F00',  # Orange
#     'quadratic': '#56B4E9',  # Sky Blue
#     'cosine': '#009E73',  # Green
# }

# # Plot evaluation results for different schedulers for all T values >= 10
# plt.figure(figsize=(12, 8))  # Larger figure size for clarity
# for scheduler_type, results in evaluation_results.items():
#     # Include all points for T >= 10 without NaN filtering
#     results = [(t, value) for t, value in results if t >= 30]
#     if results:  # Check if there are results to plot
#         T_values, metric_values = zip(*results)

#         plt.plot(
#             T_values, metric_values,
#             label=f'Scheduler: {scheduler_type}',
#             linewidth=2,
#             color=color_blind_palette.get(scheduler_type, 'black')  # Use color from the palette or fallback to black
#         )

# # Customize plot appearance
# plt.xlabel('Number of Diffusion Steps (T)', fontsize=14)
# plt.ylabel(f'{metric} Value', fontsize=14)
# plt.title(f'{metric} Between Generated and True Posterior Samples for T ≥ 10 and Different Schedulers Across Runs', fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(True)

# plt.tight_layout()

# # Save the plot as a high-resolution PDF
# plt.savefig(f'{metric.lower()}_vs_T_plot.pdf', format='pdf', dpi=300)

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import os
import sbibm
import json
import torch
from collections import defaultdict
from sbibm.metrics.c2st import c2st

# Function to compute C2ST accuracy
def compute_c2st(reference_samples, generated_samples):
    c2st_accuracy = c2st(reference_samples, generated_samples)
    return c2st_accuracy

# Directory where results are stored
results_dir = '/cephyr/users/nautiyal/Alvis/diffusion/timestep_results'

# File to save/load evaluation results
output_file = 'c2st_results.json'

# Set the specific T values to evaluate
T_values = [30, 100] + list(range(200, 1001, 100))

# Parameter to specify the number of runs
num_runs = 5  # You can change this to specify the number of runs to consider

# Check if evaluation data exists
if os.path.exists(output_file):
    # Load evaluation data from file
    with open(output_file, 'r') as f:
        evaluation_results = json.load(f)
    print(f"Evaluation results loaded from {output_file}")
else:
    # Get the task and true posterior samples
    task_name = 'two_moons'
    task = sbibm.get_task(task_name)
    true_posterior = task.get_reference_posterior_samples(num_observation=1).numpy()
    print("True posterior shape:", true_posterior.shape)

    # Convert true posterior to PyTorch tensor
    true_posterior = torch.tensor(true_posterior, dtype=torch.float32)

    # Initialize dictionary to store evaluation results
    evaluation_results = defaultdict(list)

    # Iterate through all subdirectories for different schedulers and runs
    for scheduler_type in ['linear', 'quadratic', 'cosine']:
        for T in T_values:
            metric_runs = []
            for run in range(1, num_runs + 1):  # Loop over specified number of runs
                # Path to the saved samples for the current T, scheduler, and run
                sample_path = os.path.join(results_dir, f"samples_T{T}_{scheduler_type}_run{run}.npz")
                print(f"Checking file: {sample_path}")

                if os.path.exists(sample_path):
                    print(f"File found: {sample_path}")
                    # Load generated samples
                    data = np.load(sample_path)
                    generated_samples = data['theta_samples']

                    # Convert generated samples to PyTorch tensor
                    generated_samples = torch.tensor(generated_samples, dtype=torch.float32)

                    # Compute C2ST metric
                    metric_value = compute_c2st(true_posterior, generated_samples)

                    metric_runs.append(metric_value)
                else:
                    print(f"File not found: {sample_path}")

            # If there are valid runs, calculate the mean of the metric
            if metric_runs:
                mean_metric = float(np.mean(metric_runs))  # Convert to standard float
                evaluation_results[scheduler_type].append((T, mean_metric))

    # Save results to a file
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f)
    print(f"Results saved to {output_file}")

# Define color-blind friendly colors
color_blind_palette = {
    'linear': '#E69F00',  # Orange
    'quadratic': '#56B4E9',  # Sky Blue
    'cosine': '#009E73',  # Green
}

plt.figure(figsize=(8, 6))  # More compact figure size for better readability in publications
for scheduler_type, results in evaluation_results.items():
    # Filter out T = 30 by selecting data starting from T = 100
    results = [(t, value) for t, value in results if t >= 100]
    if results:  # Check if there are results to plot
        T_values, metric_values = zip(*results)

        plt.plot(
            T_values, metric_values,
            label=f'{scheduler_type.capitalize()} Scheduler',
            linewidth=2.5,
            color=color_blind_palette.get(scheduler_type, 'black'),
            marker='o',  # Adding markers for better visual clarity
            markersize=5
        )

# Customize plot appearance
plt.xlabel(r'Number of Diffusion Steps ($T$)', fontsize=16, labelpad=10)
plt.ylabel('Mean C2ST Accuracy', fontsize=16, labelpad=10)
#plt.title('C2ST Accuracy Across Diffusion Steps', fontsize=18, pad=15)
plt.xticks(T_values, [str(t) for t in T_values], fontsize=12)
plt.yticks(np.linspace(0.5, 1.0, 6), fontsize=12)
plt.ylim(0.5, 1.0)
plt.legend(fontsize=14, loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray')
plt.grid(visible=True, which='major', linestyle='--', linewidth=0.5, color='gray')

plt.tight_layout()

# Save the plot as a high-resolution PDF
plt.savefig('c2st_accuracy_vs_T_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')

plt.show()
