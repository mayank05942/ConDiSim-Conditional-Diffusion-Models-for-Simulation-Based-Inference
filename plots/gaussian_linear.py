import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sbibm
import matplotlib.ticker as ticker
import seaborn as sns

def plot_hist(posterior_samples, real_samples=None, kde=True,
              save_path='glm.pdf'):
    # Define font sizes - increased for density and theta
    label_fontsize = 30  # Increased from 20
    theta_fontsize = 25  # New size for theta labels
    tick_fontsize = 14
    legend_fontsize = 30

    # Create DataFrames for the real and posterior samples with 10 columns
    # Use simpler column names to avoid LaTeX parsing issues
    column_names = [f'theta_{i}' for i in range(1, 11)]
    df_real = pd.DataFrame(real_samples, columns=column_names)
    df_real['Type'] = 'True Posterior'

    df_posterior = pd.DataFrame(posterior_samples, columns=column_names)
    df_posterior['Type'] = 'Generated Posterior'

    # Create the figure and GridSpec layout for 2x5 subplots
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))  # Create a 2x5 subplot

    # Define a list of colors for the histograms and KDE
    colors = ['darkred', 'darkblue']

    x_limits = {param: (-1.5, 1.5) for param in column_names}


    # Loop through each parameter to create histograms and KDE plots
    for i, param in enumerate(column_names):
        ax_hist = axes[i // 5, i % 5]

        if kde:
            # KDE plots with fill
            sns.kdeplot(data=df_real[param], ax=ax_hist, color=colors[0], 
                       label='True Posterior', linewidth=2, fill=True, alpha=0.3)
            sns.kdeplot(data=df_posterior[param], ax=ax_hist, color=colors[1], 
                       label='Generated Posterior', linewidth=2, fill=True, alpha=0.3)
        else:
            # Regular histograms
            ax_hist.hist(df_real[param], bins=50, alpha=1, color=colors[0], 
                        linewidth=1, label='True Posterior', density=True)
            ax_hist.hist(df_posterior[param], bins=100, alpha=0.4, color=colors[1], 
                        linewidth=1, label='Generated Posterior', density=True)

        # Set the labels and title with proper LaTeX formatting
        param_index = int(param.split('_')[1])
        ax_hist.set_xlabel(f'$\\theta_{{{param_index}}}$', fontsize=theta_fontsize, labelpad=10)  # Increased font and added padding
        
        # Only show y-label (Density) for the first and sixth plots (i=0 and i=5)
        if i == 0 or i == 5:
            ax_hist.set_ylabel('Density', fontsize=label_fontsize, labelpad=20)  # Added more padding
        else:
            ax_hist.set_ylabel('')

        # Set the x-axis limits using the formatted param
        ax_hist.set_xlim(x_limits[param])

        # Adjust tick label size for better visibility
        ax_hist.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # Make all border lines visible and bold
        for spine in ax_hist.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)

    plt.tight_layout(pad=1.0)

    # Create custom legend with filled patches for KDE
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
              fontsize=legend_fontsize, bbox_to_anchor=(0.5, 1.15))

    # Save the plot
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()



def generate_plots(task_name):
    # Define the base directory and absolute paths for results and plots
    base_dir = "/cephyr/users/nautiyal/Alvis/diffusion"
    results_dir = os.path.join(base_dir, "results", task_name)
    plot_dir = os.path.join(base_dir, "plots", task_name)

    # Ensure the plot directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # Check if the results directory exists
    if not os.path.exists(results_dir):
        print(f"No results found for task '{task_name}'.")
        return

    # Load the task once outside the loop for efficiency
    try:
        task = sbibm.get_task(task_name)
    except Exception as e:
        print(f"Error loading task '{task_name}': {e}")
        return

    # Get true posterior samples once
    try:
        true_posterior = task.get_reference_posterior_samples(num_observation=1)
    except Exception as e:
        print(f"Error obtaining true posterior samples for task '{task_name}': {e}")
        return

    plot_count = 0  # Counter for the number of plots generated

    for file in os.listdir(results_dir):
        # Only consider posterior files, ignoring SBC and intermediate sample files
        if file.endswith(".npz") and "run" in file and "budget" in file and "sbc" not in file and "intermediate" not in file:
            # Load posterior samples
            posterior_path = os.path.join(results_dir, file)
            try:
                data = np.load(posterior_path)
                if 'theta_samples' not in data:
                    print(f"Key 'theta_samples' not found in '{file}'. Skipping.")
                    continue
                posterior_samples = data['theta_samples']
            except Exception as e:
                print(f"Error loading '{file}': {e}")
                continue

            # Extract run and budget information from filename
            parts = file.replace('.npz', '').split('_')
            try:
                run_index = parts.index('run') + 1
                run = int(parts[run_index])
            except (ValueError, IndexError):
                print(f"Could not find 'run' in filename '{file}'. Skipping.")
                continue

            try:
                budget_index = parts.index('budget') + 1
                budget = int(parts[budget_index])
            except (ValueError, IndexError):
                print(f"Could not find 'budget' in filename '{file}'. Skipping.")
                continue

            # Define save path with the correct directory
            save_path = os.path.join(plot_dir, f"{task_name}_run_{run}_budget_{budget}.pdf")

            # Generate and save the plot
            try:
                plot_hist(posterior_samples, true_posterior, save_path=save_path)
                plot_count += 1
                print(f"Plot saved: {save_path}")
            except Exception as e:
                print(f"Error plotting '{file}': {e}")

    if plot_count == 0:
        print("No plots were generated.")
    else:
        print(f"Generated {plot_count} plot(s) in '{plot_dir}'.")

# Execute the plotting
if __name__ == "__main__":
    #task_name = "gaussian_linear"
    task_name = "gaussian_linear_uniform"
    generate_plots(task_name)
