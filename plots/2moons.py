import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
import sbibm
import re
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns


def plot_generated_vs_real_samples(real_samples, posterior_samples, kde=True, marker_size=0.1, legend_marker_size=15, save_path='comparison_plot.pdf'):
    label_fontsize = 40
    legend_fontsize = 35
    tick_fontsize = 35

    df_real = pd.DataFrame(real_samples, columns=[r'$\theta_1$', r'$\theta_2$'])
    df_real['Type'] = 'True Posterior'

    df_posterior = pd.DataFrame(posterior_samples, columns=[r'$\theta_1$', r'$\theta_2$'])
    df_posterior['Type'] = 'Generated Posterior'

    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[0.5, 0.5], figure=fig)
    plt.subplots_adjust(wspace=0.4, hspace=0.3)

    ax_scatter = fig.add_subplot(gs[:, 0])
    ax_scatter.set_facecolor('#ffffff')

    ax_scatter.scatter(df_real[r'$\theta_1$'], df_real[r'$\theta_2$'], color='darkred', s=marker_size, label='True Posterior')
    ax_scatter.scatter(df_posterior[r'$\theta_1$'], df_posterior[r'$\theta_2$'], color='darkblue', s=marker_size, label='Generated Posterior')

    ax_scatter.set_xlim(-1.1, 1.1)
    ax_scatter.set_ylim(-1.1, 1.1)



    ax_scatter.set_xlabel(r'$\theta_1$', fontsize=label_fontsize, labelpad=15)
    ax_scatter.set_ylabel(r'$\theta_2$', fontsize=label_fontsize, labelpad=15)

    ax_scatter.tick_params(axis='both', which='major', labelsize=tick_fontsize, pad=10)
    ax_scatter.xaxis.set_major_formatter(FormatStrFormatter('%.g'))
    ax_scatter.yaxis.set_major_formatter(FormatStrFormatter('%.g'))
    
    ax_scatter.set_xticks([-1, 0, 1])
    ax_scatter.set_yticks([-1, 0, 1])

    for spine in ax_scatter.spines.values():
        spine.set_linewidth(3)

    ax_hist1 = fig.add_subplot(gs[0, 1])
    sns.kdeplot(data=df_real[r'$\theta_1$'], ax=ax_hist1, color='darkred', fill=True, alpha=0.4, bw_adjust=1)
    sns.kdeplot(data=df_posterior[r'$\theta_1$'], ax=ax_hist1, color='darkblue', fill=True, alpha=0.4, bw_adjust=1)
    
    ax_hist1.set_xlabel('')
    ax_hist1.text(1.15, 0.5, r'$\theta_1$', fontsize=label_fontsize, transform=ax_hist1.transAxes, ha='left')
    ax_hist1.set_xlim(-1.5, 1.5)
    ax_hist1.tick_params(axis='both', which='major', labelsize=tick_fontsize, pad=10)
    ax_hist1.xaxis.set_major_formatter(FormatStrFormatter('%.g'))
    ax_hist1.yaxis.set_major_formatter(FormatStrFormatter('%.g'))
    ax_hist1.set_xticks([-1.5, 0, 1.5])

    ax_hist2 = fig.add_subplot(gs[1, 1])
    sns.kdeplot(data=df_real[r'$\theta_2$'], ax=ax_hist2, color='darkred', fill=True, alpha=0.4, bw_adjust=1)
    sns.kdeplot(data=df_posterior[r'$\theta_2$'], ax=ax_hist2, color='darkblue', fill=True, alpha=0.4, bw_adjust=1)
    
    ax_hist2.set_xlabel('')
    ax_hist2.text(1.15, 0.5, r'$\theta_2$', fontsize=label_fontsize, transform=ax_hist2.transAxes, ha='left')
    ax_hist2.set_xlim(-1.5, 1.5)
    ax_hist2.tick_params(axis='both', which='major', labelsize=tick_fontsize, pad=10)
    ax_hist2.xaxis.set_major_formatter(FormatStrFormatter('%.g'))
    ax_hist2.yaxis.set_major_formatter(FormatStrFormatter('%.g'))
    ax_hist2.set_xticks([-1.5, 0, 1.5])

    ax_hist1.set_ylabel("Density", fontsize=label_fontsize, labelpad=15)
    ax_hist2.set_ylabel("Density", fontsize=label_fontsize, labelpad=15)

    for ax in [ax_hist1, ax_hist2]:
        for spine in ax.spines.values():
            spine.set_linewidth(3)

    handles = [
        plt.Rectangle((0,0),1,1, fc='darkred', alpha=0.4, label='True Posterior'),
        plt.Rectangle((0,0),1,1, fc='darkblue', alpha=0.4, label='Generated Posterior')
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=legend_fontsize, 
              bbox_to_anchor=(0.55, 1.08))

    plot_dir = os.path.dirname(save_path)
    if plot_dir and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()


def generate_plots(task_name):
    base_dir = "diffusion"
    results_dir = os.path.join(base_dir, "results", task_name)
    plot_dir = os.path.join(base_dir, "plots", task_name)
    os.makedirs(plot_dir, exist_ok=True)

    if not os.path.exists(results_dir):
        print(f"No results found for task {task_name}")
        return

    pattern = re.compile(f"{task_name}_(run_\\d+_budget_\\d+|budget_\\d+_run_\\d+)\\.npz")

    for file in os.listdir(results_dir):
        if pattern.match(file):
            posterior_path = os.path.join(results_dir, file)
            data = np.load(posterior_path)
            posterior_samples = data['theta_samples']

            task = sbibm.get_task(task_name)
            true_posterior = task.get_reference_posterior_samples(num_observation=1)

            parts = file.split('_')
            try:
                run_index = parts.index('run') + 1
                run = int(parts[run_index])

                budget_index = parts.index('budget') + 1
                budget_str = parts[budget_index]
                budget = int(budget_str.split('.')[0])
            except (ValueError, IndexError):
                print(f"Unexpected filename format for {file}; could not extract 'run' or 'budget'.")
                continue

            save_path = os.path.join(plot_dir, f"{task_name}_run_{run}_budget_{budget}.pdf")

            plot_generated_vs_real_samples(true_posterior, posterior_samples, save_path=save_path)


if __name__ == "__main__":
    task_name = "two_moons"
    generate_plots(task_name)
