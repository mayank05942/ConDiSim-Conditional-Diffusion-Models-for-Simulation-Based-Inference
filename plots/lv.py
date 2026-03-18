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
import seaborn as sns

def plot_hist(posterior_samples, real_samples=None, save_path='lv_post_v2.pdf'):
    label_fontsize = 40
    param_fontsize = 45
    tick_fontsize = 25
    legend_fontsize = 35

    df_real = pd.DataFrame(real_samples, columns=[r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$']) if real_samples is not None else None
    df_posterior = pd.DataFrame(posterior_samples, columns=[r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$'])

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    plt.subplots_adjust(wspace=0.4)  # Increased spacing between subplots
    axes = axes.flatten()

    colors = ['darkred', 'darkblue']

    x_limits = {
        r'$\alpha$': (0, 2),
        r'$\beta$': (0, 1),
        r'$\gamma$': (0, 2),
        r'$\delta$': (0, 1)
    }

    # Loop through each parameter to create plots
    for i, param in enumerate([r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$']):
        ax = axes[i]

        # Plot histogram for generated posterior
        counts_posterior, bins_posterior = np.histogram(df_posterior[param], bins=50)
        ax.hist(df_posterior[param], bins=50, alpha=0.4, color=colors[1],
                histtype='stepfilled', linewidth=2, label='Generated Posterior')
        
        # Plot vertical line for true posterior (mean value)
        true_mean = np.mean(df_real[param])
        max_height = max(counts_posterior) * 1.1
        ax.vlines(true_mean, 0, max_height, color=colors[0], linewidth=3, label='True Posterior')


        ax.set_xlabel(param, fontsize=param_fontsize, labelpad=15)
        if i == 0:
            ax.set_ylabel('Frequency', fontsize=label_fontsize, labelpad=20)
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])

        ax.set_xlim(x_limits[param])

        ax.set_xticks([xmin, (xmin + xmax)/2, xmax])
        
        xmin, xmax = x_limits[param]
        ax.set_ylim(0, max_height)
        
        ax.yaxis.set_major_locator(MaxNLocator(4))
        
        def format_ticks(x, p):
            s = f'{x:.1f}'
            return s.rstrip('0').rstrip('.') if '.' in s else s
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_ticks))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_ticks))
        
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, pad=8)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)

    # Create custom legend
    handles = [
        plt.Rectangle((0,0),1,1, fc='darkred', alpha=0.4, label='True Posterior'),
        plt.Rectangle((0,0),1,1, fc='darkblue', alpha=0.4, label='Generated Posterior')
    ]
    fig.legend(handles=handles, loc='upper center', fontsize=legend_fontsize, 
              bbox_to_anchor=(0.5, 1.25), ncol=2)

    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=600)

    plt.show()



def generate_plots(task_name):
    base_dir = "diffusion"
    results_dir = os.path.join(base_dir, "results", task_name)
    
    if not os.path.exists(results_dir):
        print(f"No results found for task {task_name} at {results_dir}")
        return
        
    plot_dir = os.path.join(base_dir, "plots", task_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Looking for results in: {results_dir}")
    print(f"Will save plots to: {plot_dir}")

    for file in os.listdir(results_dir):
        if file.endswith(".npz") and "run" in file and "budget" in file and "sbc" not in file and "intermediate" not in file:
            posterior_path = os.path.join(results_dir, file)
            try:
                data = np.load(posterior_path)
                if 'theta_samples' not in data:
                    print(f"Key 'theta_samples' not found in file: {file}")
                    continue
                posterior_samples = data['theta_samples']
            except Exception as e:
                print(f"Error loading file '{file}': {e}")
                continue

            task = sbibm.get_task(task_name)
            true_posterior = task.get_reference_posterior_samples(num_observation=1)

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

# Execute the plotting
if __name__ == "__main__":
    task_name = "lotka_volterra"

    generate_plots(task_name)
