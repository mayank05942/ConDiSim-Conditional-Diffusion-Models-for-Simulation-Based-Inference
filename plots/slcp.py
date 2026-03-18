import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sbibm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import re
import seaborn as sns

def plot_generated_vs_real_samples(real_samples, posterior_samples, legend_marker_size=15, save_path='comparison_plot.pdf'):
    columns = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$', r'$\theta_5$']

    df_real = pd.DataFrame(real_samples, columns=columns)
    df_real['Type'] = 'True Posterior'

    df_posterior = pd.DataFrame(posterior_samples, columns=columns)
    df_posterior['Type'] = 'Generated Posterior'

    label_fontsize = 60
    tick_fontsize = 35
    legend_fontsize = 50

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    plt.subplots_adjust(wspace=0.3)

    x_limits = {
        r'$\theta_1$': (-5, 5),
        r'$\theta_2$': (-2, 2),
        r'$\theta_3$': (-4, 4),
        r'$\theta_4$': (-3, 3),
        r'$\theta_5$': (-1, 3.5),
    }

    for i, col in enumerate(columns):
        sns.kdeplot(data=df_real[col], ax=axes[i], color='darkred', alpha=0.4, fill=True, label='True Posterior')
        sns.kdeplot(data=df_posterior[col], ax=axes[i], color='darkblue', alpha=0.4, fill=True, label='Generated Posterior')

        axes[i].set_xlabel(col, fontsize=label_fontsize, labelpad=15)
        axes[i].tick_params(axis='both', which='major', labelsize=tick_fontsize)

        if i != 0:
            axes[i].set_ylabel('')

        axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.g'))
        axes[i].tick_params(axis='y', pad=15)

        # Format x-axis ticks to remove decimal from 0
        axes[i].xaxis.set_major_formatter(FormatStrFormatter('%.g'))

        # Set x-limits if defined
        if col in x_limits:
            axes[i].set_xlim(x_limits[col])

        # Make the plot frame borders bolder
        for spine in axes[i].spines.values():
            spine.set_linewidth(3)

    # Set Y-axis label only for the first subplot with padding
    axes[0].set_ylabel('Density', fontsize=label_fontsize, labelpad=20)

    # Create custom legend and place it outside the plot area
    handles = [
        plt.Rectangle((0,0),1,1, fc='darkred', alpha=0.4, label='True Posterior'),
        plt.Rectangle((0,0),1,1, fc='darkblue', alpha=0.4, label='Generated Posterior')
    ]
    fig.legend(handles=handles, loc='upper center', fontsize=legend_fontsize, 
              bbox_to_anchor=(0.5, 1.25), ncol=2)

    # Save the plot as a PDF file with higher DPI
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=600)

    # Show the plot
    plt.show()



def generate_plots(task_name):
    # Use absolute paths to ensure plots are saved in the correct location
    base_dir = "diffusion"
    results_dir = os.path.join(base_dir, "results", task_name)
    if not os.path.exists(results_dir):
        print(f"No results found for task {task_name}")
        return

    plot_dir = os.path.join(base_dir, "plots", task_name)
    os.makedirs(plot_dir, exist_ok=True)  # Ensure that plot directory exists

    for file in os.listdir(results_dir):
        # Only process files that match the posterior filename pattern (e.g., no intermediate or SBC files)
        if file.endswith(".npz") and "run" in file and "budget" in file and "sbc" not in file and "intermediate" not in file:
            # Load posterior samples
            posterior_path = os.path.join(results_dir, file)
            try:
                data = np.load(posterior_path)
                posterior_samples = data['theta_samples']
            except KeyError:
                print(f"Key 'theta_samples' not found in file: {file}")
                continue

            # Load true posterior samples
            task = sbibm.get_task(task_name)
            true_posterior = task.get_reference_posterior_samples(num_observation=1)

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

            # Define the path to save the plot in the correct directory
            save_path = os.path.join(plot_dir, f"{task_name}_run_{run}_budget_{budget}.pdf")

            # Generate plot
            try:
                plot_generated_vs_real_samples(
                    real_samples=true_posterior,
                    posterior_samples=posterior_samples,
                    legend_marker_size=15,
                    save_path=save_path
                )
                print(f"Plot saved: {save_path}")
            except Exception as e:
                print(f"Error plotting '{file}': {e}")


# Execute the plotting
if __name__ == "__main__":
    #task_name = "slcp"
    task_name = "slcp_distractors"
    generate_plots(task_name)
