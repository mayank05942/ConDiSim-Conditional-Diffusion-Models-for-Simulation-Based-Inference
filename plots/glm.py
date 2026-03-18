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
    label_fontsize = 30
    theta_fontsize = 25
    tick_fontsize = 14
    legend_fontsize = 30

    column_names = [f'theta_{i}' for i in range(1, 11)]
    df_real = pd.DataFrame(real_samples, columns=column_names)
    df_real['Type'] = 'True Posterior'

    df_posterior = pd.DataFrame(posterior_samples, columns=column_names)
    df_posterior['Type'] = 'Generated Posterior'

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    colors = ['darkred', 'darkblue']

    x_limits = {
        'theta_1': (-1, 3),
        'theta_2': (0.5, 4.0),
        'theta_3': (1, 6),
        'theta_4': (0.5, 5),
        'theta_5': (-1, 2),
        'theta_6': (-3, 0),
        'theta_7': (-4.5, -1),
        'theta_8': (-3.5, 0),
        'theta_9': (-2, 1),
        'theta_10': (-1, 2),
    }

    for i, param in enumerate(column_names):
        ax_hist = axes[i // 5, i % 5]

        if kde:
            # KDE plots with fill
            sns.kdeplot(data=df_real[param], ax=ax_hist, color=colors[0], 
                       label='True Posterior', linewidth=2, fill=True, alpha=0.3)
            sns.kdeplot(data=df_posterior[param], ax=ax_hist, color=colors[1], 
                       label='Generated Posterior', linewidth=2, fill=True, alpha=0.3)
        else:
            ax_hist.hist(df_real[param], bins=50, alpha=1, color=colors[0], 
                        linewidth=1, label='True Posterior', density=True)
            ax_hist.hist(df_posterior[param], bins=100, alpha=0.4, color=colors[1], 
                        linewidth=1, label='Generated Posterior', density=True)

        param_index = int(param.split('_')[1])
        ax_hist.set_xlabel(f'$\\theta_{{{param_index}}}$', fontsize=theta_fontsize, labelpad=10)  # Increased font and added padding
        
        if i == 0 or i == 5:
            ax_hist.set_ylabel('Density', fontsize=label_fontsize, labelpad=20)  # Added more padding
        else:
            ax_hist.set_ylabel('')

        ax_hist.set_xlim(x_limits[param])

        ax_hist.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        for spine in ax_hist.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)

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
              fontsize=legend_fontsize, bbox_to_anchor=(0.5, 1.15))

    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()


def generate_plots(task_name):
    base_dir = "diffusion"
    results_dir = os.path.join(base_dir, "results", task_name)
    plot_dir = os.path.join(base_dir, "plots", task_name)

    os.makedirs(plot_dir, exist_ok=True)

    if not os.path.exists(results_dir):
        print(f"No results found for task '{task_name}'.")
        return

    try:
        task = sbibm.get_task(task_name)
    except Exception as e:
        print(f"Error loading task '{task_name}': {e}")
        return

    try:
        true_posterior = task.get_reference_posterior_samples(num_observation=1)
    except Exception as e:
        print(f"Error obtaining true posterior samples for task '{task_name}': {e}")
        return

    plot_count = 0

    for file in os.listdir(results_dir):
        if file.endswith(".npz") and "run" in file and "budget" in file and "sbc" not in file and "intermediate" not in file:
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

            save_path = os.path.join(plot_dir, f"{task_name}_run_{run}_budget_{budget}.pdf")

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

if __name__ == "__main__":
    task_name = "bernoulli_glm_raw"
    generate_plots(task_name)
