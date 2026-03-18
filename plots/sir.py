import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
from matplotlib.lines import Line2D
import sbibm
import torch
import seaborn as sns

def plot_hist(posterior_samples, real_samples=None, kde=True,
              save_path='sir_v1_post.pdf'):
    posterior_samples = posterior_samples.numpy() if isinstance(posterior_samples, torch.Tensor) else posterior_samples
    real_samples = real_samples.numpy() if isinstance(real_samples, torch.Tensor) else real_samples

    if posterior_samples.shape[1] != 2 or real_samples.shape[1] != 2:
        raise ValueError("Both posterior_samples and real_samples must have exactly two columns (for alpha and beta).")

    df_real = pd.DataFrame(real_samples, columns=[r'$\alpha$', r'$\beta$'])
    df_real['Type'] = 'True Posterior'

    df_posterior = pd.DataFrame(posterior_samples, columns=[r'$\alpha$', r'$\beta$'])
    df_posterior['Type'] = 'Generated Posterior'

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['darkred', 'darkblue']

    x_limits = {
        r'$\alpha$': (0.55, 0.70),
        r'$\beta$': (0.10, 0.25),
    }

    label_fontsize = 35
    param_fontsize = 40
    tick_fontsize = 14
    legend_fontsize = 30

    # Loop through each parameter to create KDE plots
    for i, param in enumerate([r'$\alpha$', r'$\beta$']):
        ax_hist = axes[i]

        if kde:
            sns.kdeplot(data=df_real[param], ax=ax_hist, color=colors[0], 
                       label='True Posterior', linewidth=2.5, fill=True, alpha=0.4)
            sns.kdeplot(data=df_posterior[param], ax=ax_hist, color=colors[1], 
                       label='Generated Posterior', linewidth=2.5, fill=True, alpha=0.4)
        else:
            ax_hist.hist(df_real[param], bins=50, alpha=1, color=colors[0], 
                        linewidth=1, label='True Posterior', density=True)
            ax_hist.hist(df_posterior[param], bins=50, alpha=0.4, color=colors[1], 
                        linewidth=1, label='Generated Posterior', density=True)

        ax_hist.set_xlim(x_limits[param])

        if param == r'$\alpha$':
            ax_hist.set_xticks([0.55, 0.60, 0.65, 0.70])
        else:
            ax_hist.set_xticks([0.10, 0.15, 0.20, 0.25])

        ax_hist.set_xlabel(param, fontsize=param_fontsize, labelpad=10)

        ax_hist.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        for spine in ax_hist.spines.values():
            spine.set_linewidth(2)

        for spine in ax_hist.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)

    axes[0].set_ylabel('Density', fontsize=label_fontsize, labelpad=30)
    
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        yticks = np.linspace(ymin, ymax, 4)[1:]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{int(y)}" for y in yticks])
    axes[1].set_ylabel('')

    plt.tight_layout(pad=1.0)

    if kde:
        handles = [
            plt.Rectangle((0,0), 1, 1, fc=colors[0], alpha=0.3, label='True Posterior'),
            plt.Rectangle((0,0), 1, 1, fc=colors[1], alpha=0.3, label='Generated Posterior')
        ]
    else:
        handles = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[0], 
                      markersize=10, label='True Posterior'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[1], 
                      markersize=10, label='Generated Posterior')
        ]

    fig.legend(handles=handles, loc='upper center', ncol=2, 
              fontsize=legend_fontsize, bbox_to_anchor=(0.53, 1.2))

    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

# def plot_hist(posterior_samples, real_samples=None, kde=True, save_path='sir_v2_post.pdf'):
#     """
#     Plots histograms or KDE plots for posterior and real samples.

#     Args:
#         posterior_samples (array-like): Samples generated from the posterior distribution.
#         real_samples (array-like, optional): True posterior samples for comparison. Default is None.
#         kde (bool, optional): If True, uses KDE plots instead of histograms. Default is False.
#         save_path (str, optional): Path to save the plot. Default is 'sir_v2_post.pdf'.
#     """
#     # Convert torch tensors to numpy arrays if necessary
#     posterior_samples = posterior_samples.numpy() if isinstance(posterior_samples, torch.Tensor) else posterior_samples
#     real_samples = real_samples.numpy() if isinstance(real_samples, torch.Tensor) else real_samples

#     # Check if the input samples have 2 columns (for alpha and beta)
#     if posterior_samples.shape[1] != 2 or (real_samples is not None and real_samples.shape[1] != 2):
#         raise ValueError("Both posterior_samples and real_samples must have exactly two columns (for alpha and beta).")

#     # Create DataFrames for the real and posterior samples with 2 columns
#     if real_samples is not None:
#         df_real = pd.DataFrame(real_samples, columns=[r'$\alpha$', r'$\beta$'])
#         df_real['Type'] = 'True Posterior'

#     df_posterior = pd.DataFrame(posterior_samples, columns=[r'$\alpha$', r'$\beta$'])
#     df_posterior['Type'] = 'Generated Posterior'

#     # Create the figure and GridSpec layout
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Create a 1x2 subplot

#     # Define a list of colors for the histograms or KDE
#     #colors = ['salmon', 'blue']
#     colors = ['darkred', 'darkblue']

#     # Define x-axis limits based on the distribution ranges
#     x_limits = {
#         r'$\alpha$': (0.55, 0.70),
#         r'$\beta$': (0.10, 0.25),
#     }

#     # Adjust font sizes for clarity
#     label_fontsize = 20
#     legend_fontsize = 20
#     tick_fontsize = 14

#     # Loop through each parameter to create plots
#     for i, param in enumerate([r'$\alpha$', r'$\beta$']):
#         ax = axes[i]

#         if kde:
#             # Plot KDE for real samples if available
#             if real_samples is not None:
#                 sns.kdeplot(df_real[param], ax=ax, color=colors[0], fill=True, label='True Posterior', alpha=0.5, bw_adjust=1.5)

#             # Plot KDE for posterior samples
#             sns.kdeplot(df_posterior[param], ax=ax, color=colors[1], fill=True, label='Generated Posterior', alpha=0.4, bw_adjust=1.5)
#         else:
#             # Plot histogram for the real samples with higher line width for better visibility
#             if real_samples is not None:
#                 ax.hist(df_real[param], bins=50, alpha=1, color=colors[0], linewidth=1, label='True Posterior')

#             # Plot histogram for the posterior samples with transparency
#             ax.hist(df_posterior[param], bins=50, alpha=0.4, color=colors[1], linewidth=1, label='Generated Posterior')

#         # Set the x-axis limits
#         ax.set_xlim(x_limits[param])

#         # Set x-ticks (reduce the number of ticks)
#         if param == r'$\alpha$':
#             ax.set_xticks([0.55, 0.60, 0.65, 0.70])  # Adjust for alpha
#         else:
#             ax.set_xticks([0.10, 0.15, 0.20, 0.25])  # Adjust for beta

#         # Set the x-axis label with larger font size and bold math symbols
#         ax.set_xlabel(param, fontsize=label_fontsize, fontweight='bold')

#         # Increase tick label size for better visibility
#         ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

#         # Remove individual subplot Y-axis labels
#         ax.set_ylabel("")

#         # Make the border lines bolder for the plots
#         for spine in ax.spines.values():
#             spine.set_linewidth(2)

#     # Add a common Y-axis label dynamically (Frequency or Density)
#     y_label = 'Density' if kde else 'Frequency'
#     fig.text(0.04, 0.5, y_label, va='center', rotation='vertical', fontsize=label_fontsize)

#     # Create custom legend with square markers
#     handles = [
#         plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[0], markersize=10, label='True Posterior'),
#         plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[1], markersize=10, label='Generated Posterior')
#     ]
#     fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=legend_fontsize, bbox_to_anchor=(0.5, 1.05))

#     # Save the plot
#     plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)

#     # Show the plot
#     plt.show()


# def plot_hist(posterior_samples, real_samples=None, kde=False,
#               save_path='sir_v2_post.pdf'):
#     # Convert torch tensors to numpy arrays if necessary
#     posterior_samples = posterior_samples.numpy() if isinstance(posterior_samples, torch.Tensor) else posterior_samples
#     real_samples = real_samples.numpy() if isinstance(real_samples, torch.Tensor) else real_samples

#     # Check if the input samples have 2 columns (for alpha and beta)
#     if posterior_samples.shape[1] != 2 or real_samples.shape[1] != 2:
#         raise ValueError("Both posterior_samples and real_samples must have exactly two columns (for alpha and beta).")

#     # Create DataFrames for the real and posterior samples with 2 columns
#     df_real = pd.DataFrame(real_samples, columns=[r'$\alpha$', r'$\beta$'])
#     df_real['Type'] = 'True Posterior'

#     df_posterior = pd.DataFrame(posterior_samples, columns=[r'$\alpha$', r'$\beta$'])
#     df_posterior['Type'] = 'Generated Posterior'

#     # Create the figure and GridSpec layout
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Create a 1x2 subplot

#     # Define a list of colors for the histograms
#     colors = ['salmon', 'blue']

#     # Define x-axis limits based on the distribution ranges
#     # x_limits = {
#     #     r'$\alpha$': (0.4, 0.8),
#     #     r'$\beta$': (0.1, 0.3),
#     # }

#     x_limits = {
#         r'$\alpha$': (0.55, 0.70),
#         r'$\beta$': (0.10, 0.25),
#     }

#     # Adjust font sizes for clarity
#     label_fontsize = 20
#     legend_fontsize = 20
#     tick_fontsize = 14

#     # Loop through each parameter to create histograms
#     for i, param in enumerate([r'$\alpha$', r'$\beta$']):
#         ax_hist = axes[i]

#         # Plot histogram for the real samples with higher line width for better visibility
#         ax_hist.hist(df_real[param], bins=50, alpha=1, color=colors[0], linewidth=1, label='True Posterior')

#         # Plot histogram for the posterior samples with transparency
#         ax_hist.hist(df_posterior[param], bins=50, alpha=0.4, color=colors[1], linewidth=1, label='Generated Posterior')

#         # Set the x-axis limits
#         ax_hist.set_xlim(x_limits[param])

#         #Set x-ticks (reduce the number of ticks)
#         if param == r'$\alpha$':
#             ax_hist.set_xticks([0.55, 0.60, 0.65, 0.70])  # Adjust for alpha
#         else:
#             ax_hist.set_xticks([0.10, 0.15, 0.20, 0.25])  # Adjust for beta

#         # Set the x-axis label with larger font size and bold math symbols
#         ax_hist.set_xlabel(param, fontsize=label_fontsize, fontweight='bold')

#         # Increase tick label size for better visibility
#         ax_hist.tick_params(axis='both', which='major', labelsize=tick_fontsize)

#         # Adjust y-axis limits and set intervals
#         # ax_hist.set_ylim(0, 1200)
#         # ax_hist.set_yticks(range(0, 1400, 200))

#         # Make the border lines bolder for the histograms
#         for spine in ax_hist.spines.values():
#             spine.set_linewidth(2)

#     # Add a common Y-axis label (Frequency) in the center of the left subplot
#     fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=label_fontsize)

#     # Create custom legend
#     handles = [
#         plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[0], markersize=10, label='True Posterior'),
#         plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[1], markersize=10, label='Generated Posterior')
#     ]
#     fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=legend_fontsize, bbox_to_anchor=(0.5, 1.05))


#     plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)

#     # Show the plot
#     plt.show()

def generate_plots(task_name):
    # Use absolute paths to ensure plots are saved in the correct location
    base_dir = "diffusion"
    results_dir = os.path.join(base_dir, "results", task_name)
    
    # Check if the results directory exists
    if not os.path.exists(results_dir):
        print(f"No results found for task {task_name} at {results_dir}")
        return
        
    plot_dir = os.path.join(base_dir, "plots", task_name)
    os.makedirs(plot_dir, exist_ok=True)  # Ensure that plot directory exists
    
    print(f"Looking for results in: {results_dir}")
    print(f"Will save plots to: {plot_dir}")

    # Check if results directory is empty
    files_found = [f for f in os.listdir(results_dir)
                  if f.endswith(".npz") and "run" in f and "budget" in f
                  and "sbc" not in f and "intermediate" not in f]

    if not files_found:
        print(f"No matching .npz files found in {results_dir}")
        print(f"Available files: {os.listdir(results_dir)}")
        return

    for file in files_found:
        # Load posterior samples
        posterior_path = os.path.join(results_dir, file)
        try:
            data = np.load(posterior_path)
            if 'theta_samples' not in data:
                print(f"Key 'theta_samples' not found in file: {file}")
                continue
            posterior_samples = data['theta_samples']
            print(f"Loaded {len(posterior_samples)} posterior samples from {file}")
        except Exception as e:
            print(f"Error loading file '{file}': {e}")
            continue

        # Load true posterior samples
        task = sbibm.get_task(task_name)
        true_posterior = task.get_reference_posterior_samples(num_observation=1)
        print(f"Loaded {len(true_posterior)} true posterior samples")

        # Extract run and budget information from filename
        parts = file.split('_')
        try:
            run_index = parts.index('run') + 1
            run = int(parts[run_index])

            budget_index = parts.index('budget') + 1
            budget = int(parts[budget_index].split('.')[0])
        except (ValueError, IndexError):
            print(f"Unexpected filename format; could not extract 'run' or 'budget' for file: {file}")
            continue

        # Generate plot and save it to the correct directory
        save_path = os.path.join(plot_dir, f"{task_name}_run_{run}_budget_{budget}.pdf")
        try:
            print(f"Generating plot for {file} with {len(posterior_samples)} posterior samples and {len(true_posterior)} true samples")
            plot_hist(
                posterior_samples=posterior_samples,
                real_samples=true_posterior,
                save_path=save_path
            )
            print(f"Plot saved: {save_path}")
        except Exception as e:
            print(f"Error generating plot for file '{file}': {e}")
            continue

# Execute the plotting
if __name__ == "__main__":
    task_name = "sir"
    
    generate_plots(task_name)
