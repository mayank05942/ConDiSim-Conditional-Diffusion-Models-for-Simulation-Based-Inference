import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy.stats import gaussian_kde
from vilar_dataset import simulator, Vilar_Oscillator, parameter_names
from gillespy2 import SSACSolver


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 20
plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern math font without LaTeX

from matplotlib.ticker import FuncFormatter

def clean_formatter(x, pos):
    if abs(x) < 1e-10:
        return '0'
    elif abs(x - round(x)) < 1e-10:
        return f'{int(round(x))}'
    elif abs(x) < 0.01:
        return f'{x:.3f}'.rstrip('0').rstrip('.')
    elif abs(x) < 1:
        return f'{x:.2f}'.rstrip('0').rstrip('.')
    elif abs(x) < 100:
        return f'{x:.1f}'.rstrip('0').rstrip('.')
    else:
        return f'{int(round(x))}'

def plot_posterior_distributions(posterior_samples, true_theta, save_path):
    param_names = [
        r'$\alpha_a$', r'$\alpha_a^\prime$', r'$\alpha_r$', r'$\alpha_r^\prime$',
        r'$\beta_a$', r'$\beta_r$', r'$\delta_{ma}$', r'$\delta_{mr}$', r'$\gamma_a$',
        r'$\gamma_r$', r'$\gamma_c$', r'$\theta_a$', r'$\theta_r$', r'$\delta_a$', r'$\delta_r$'
    ]
    
    prior_ranges = [
        (0, 100),
        (0, 1000),
        (0, 10),
        (0, 100),
        (0, 100),
        (0, 10),
        (0, 20),
        (0, 4),
        (0, 4),
        (0, 2),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 100),
        (0, 200)
    ]

    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(18, 10.8))
    
    for i in range(15):
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        
        samples_i = posterior_samples[:, i]
        p_low = np.percentile(samples_i, 0.5)
        p_high = np.percentile(samples_i, 99.5)
        samples_filtered = samples_i[(samples_i >= p_low) & (samples_i <= p_high)]
        
        kde = sns.kdeplot(
            x=samples_filtered,
            ax=ax,
            color='#1f77b4',
            fill=True,
            alpha=0.6,
            bw_adjust=0.5
        )
        
        ax.set_xlim(prior_ranges[i])
        
        true_val = true_theta[i]
        ax.axvline(true_val, color='#d62728', linestyle='--', linewidth=4)
        
        current_xlim = ax.get_xlim()
        if true_val < current_xlim[0] or true_val > current_xlim[1]:
            new_min = min(current_xlim[0], true_val - 0.1 * abs(true_val))
            new_max = max(current_xlim[1], true_val + 0.1 * abs(true_val))
            ax.set_xlim(new_min, new_max)
        
        ax.set_ylim(bottom=0)
        
        ax.set_xlabel('')
        ax.text(0.5, 1.08, param_names[i], fontsize=35, fontweight='bold',
                ha='center', va='bottom', transform=ax.transAxes)
        
        if col == 0:
            if row == 0:
                ax.set_ylabel('Density', fontsize=30, labelpad=10)
            else:
                ax.set_ylabel('Density', fontsize=30, labelpad=25)
        else:
            ax.set_ylabel('')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        ax.xaxis.set_major_formatter(FuncFormatter(clean_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(clean_formatter))
        
        current_xticks = list(ax.get_xticks())
        if true_val not in current_xticks:
            current_xticks.append(true_val)
            current_xticks.sort()
            ax.set_xticks(current_xticks)
        
        ax.tick_params(axis='both', which='major', labelsize=18, width=1.5)
        
        for idx, label in enumerate(ax.get_xticklabels()):
            try:
                val = float(label.get_text().replace('−', '-'))
                if 0 < val < 1 and val < xlim[1] * 0.15:
                    if val <= 0.02:
                        from matplotlib.transforms import ScaledTranslation
                        offset = ScaledTranslation(30/72., 0, ax.figure.dpi_scale_trans)
                        label.set_transform(label.get_transform() + offset)
                    else:
                        label.set_horizontalalignment('left')
            except:
                pass

    handles = [
        plt.Line2D([0], [0], color='#1f77b4', lw=10, alpha=0.6),
        plt.Line2D([0], [0], color='#d62728', lw=4, linestyle='--')
    ]
    labels = ['Generated Posterior', 'True Parameter']
    legend = fig.legend(handles, labels, loc='upper center', 
              bbox_to_anchor=(0.5, 1.02), ncol=2, 
              fontsize=22, frameon=True, fancybox=False)
    legend.get_frame().set_linewidth(2)
    legend.get_frame().set_edgecolor('black')

    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close()



def plot_posterior_predictive(posterior_samples, observed_data, save_path):
    model = Vilar_Oscillator()
    solver = SSACSolver(model=model)
    
    np.random.seed(42)
    num_samples = 3
    random_indices = np.random.choice(len(posterior_samples), num_samples, replace=False)
    selected_samples = posterior_samples[random_indices]
    
    simulated_data = []
    for params in selected_samples:
        sim_result = simulator(params[np.newaxis, :], model, solver, transform=True)
        sim_result = sim_result.squeeze(0)
        simulated_data.append(sim_result)
    simulated_data = np.array(simulated_data)
    
    species_names = ['Complex (C)', 'Activator (A)', 'Repressor (R)']
    colors = ['#2E86C1', '#E74C3C', '#2ECC71']
    time = np.arange(observed_data.shape[1])
    
    fig, axes = plt.subplots(3, 1, figsize=(7, 8))
    plt.subplots_adjust(hspace=0.4)
    
    for i, (ax, species, color) in enumerate(zip(axes, species_names, colors)):
        true_line = ax.plot(
            time,
            observed_data[i],
            color=color,
            linewidth=1.5,
            label='True',
            zorder=5
        )[0]
        
        gen_line = None
        for j in range(num_samples):
            line = ax.plot(
                time,
                simulated_data[j, i],
                alpha=0.3,
                color=color,
                linewidth=0.8,
                label='Generated' if j == 0 else None
            )
            if j == 0:
                gen_line = line[0]
        
        ax.set_ylabel('Population', fontsize=12)
        if i == 2:
            ax.set_xlabel('Time', fontsize=12)
        
        ax.text(
            0.3, 1.05, species,
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            ha='center'
        )
        
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        leg = ax.legend(
            [true_line, gen_line],
            ['True', 'Generated'],
            fontsize=10,
            loc='upper center',
            bbox_to_anchor=(0.8, 1.2),
            frameon=True,
            framealpha=0.9,
            ncol=2
        )
        
        for line in leg.get_lines():
            line.set_linewidth(4)
        
        leg.get_frame().set_linewidth(0.5)
    
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
def main():
    os.makedirs('vilar_plots', exist_ok=True)
    for file_name in os.listdir('posterior_samples'):
        if not file_name.endswith('.npz'):
            continue

        print(f"Processing {file_name}")
        file_path = os.path.join('posterior_samples', file_name)
        data = np.load(file_path, allow_pickle=True)
        
        match = re.search(r'budget(\d+)', file_name)
        if match:
            budget = match.group(1)
        else:
            budget = "unknown"
        
        if 'theta_samples' in data:
            samples_key = 'theta_samples'
        elif 'posterior_samples' in data:
            samples_key = 'posterior_samples'
        else:
            print(f"Warning: No samples found in {file_name}. Available keys: {list(data.keys())}")
            continue
        
        posterior_samples = data[samples_key]
        
        if 'true_theta' in data:
            true_params = data['true_theta']
        elif 'true_parameters' in data:
            true_params = data['true_parameters']
        else:
            print(f"Warning: No true parameters found in {file_name}. Available keys: {list(data.keys())}")
            continue
        
        plot_base_name = f'vilar_plots/{budget}'
        
        plot_posterior_distributions(
            posterior_samples,
            true_params,
            f'{plot_base_name}_dist.pdf'
        )
        
        dataset_path = os.path.join('datasets', f'vilar_dataset_{budget}.npz')
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset file not found: {dataset_path}")
            continue
        
        dataset = np.load(dataset_path, allow_pickle=True)
        
        if 'true_ts' in dataset:
            ts_data = dataset['true_ts']
        else:
            print(f"Warning: 'true_ts' not found in dataset file. Available keys: {list(dataset.keys())}")
            continue
        
        plot_posterior_predictive(
            posterior_samples,
            ts_data,
            f'{plot_base_name}_pred.pdf'
        )
        
        print(f"Saved plots for budget {budget}")


if __name__ == "__main__":
    main()
