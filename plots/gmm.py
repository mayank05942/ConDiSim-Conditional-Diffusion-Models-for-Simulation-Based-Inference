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
              save_path='gmm_post.pdf'):
    posterior_samples = posterior_samples.numpy() if isinstance(posterior_samples, torch.Tensor) else posterior_samples
    real_samples = real_samples.numpy() if isinstance(real_samples, torch.Tensor) else real_samples

    if real_samples is not None:
        df_real = pd.DataFrame(real_samples, columns=[r'$\theta_1$', r'$\theta_2$'])
        df_real['Type'] = 'True Posterior'
    else:
        df_real = pd.DataFrame(columns=[r'$\theta_1$', r'$\theta_2$', 'Type'])

    df_posterior = pd.DataFrame(posterior_samples, columns=[r'$\theta_1$', r'$\theta_2$'])
    df_posterior['Type'] = 'Generated Posterior'

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['darkred', 'darkblue']

    x_limits = {
        r'$\theta_1$': (-10.1, -7),
        r'$\theta_2$': (-4.0, 1),
    }

    label_fontsize = 35
    param_fontsize = 40
    tick_fontsize = 14
    legend_fontsize = 30

    for i, param in enumerate([r'$\theta_1$', r'$\theta_2$']):
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
        
        ax_hist.xaxis.set_major_locator(MaxNLocator(integer=True))
        xticks = ax_hist.get_xticks()
        ax_hist.set_xticks(xticks)
        ax_hist.set_xticklabels([f"{int(tick)}" for tick in xticks])

        ax_hist.set_xlabel(param, fontsize=param_fontsize, labelpad=10)

        ax_hist.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        for spine in ax_hist.spines.values():
            spine.set_linewidth(2)

        for spine in ax_hist.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)

    axes[0].set_ylabel('Density', fontsize=label_fontsize, labelpad=30)
    
    for ax in axes:
        start, end = ax.get_ylim()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ticks = [t for t in ax.get_yticks() if t != 0]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{int(tick)}" for tick in ticks])
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
                print(f"Error plotting '{file}': {e}")

if __name__ == "__main__":
    task_name = "gaussian_mixture"
    generate_plots(task_name)
