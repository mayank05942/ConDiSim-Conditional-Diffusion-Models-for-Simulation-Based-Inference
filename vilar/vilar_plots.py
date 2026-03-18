import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')  # must be before pyplot import
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

# Custom formatter to avoid scientific notation and clean up zeros
from matplotlib.ticker import FuncFormatter

def clean_formatter(x, pos):
    """Format tick labels to show clean integers and avoid 0.0, 0.00, etc."""
    if abs(x) < 1e-10:  # Treat very small numbers as zero
        return '0'
    elif abs(x - round(x)) < 1e-10:  # Close to integer
        return f'{int(round(x))}'
    elif abs(x) < 0.01:  # Very small numbers
        return f'{x:.3f}'.rstrip('0').rstrip('.')
    elif abs(x) < 1:  # Numbers between 0 and 1
        return f'{x:.2f}'.rstrip('0').rstrip('.')
    elif abs(x) < 100:  # Regular numbers
        return f'{x:.1f}'.rstrip('0').rstrip('.')
    else:  # Large numbers
        return f'{int(round(x))}'

def plot_posterior_distributions(posterior_samples, true_theta, save_path):
    """Create KDE plots for posterior distributions of parameters."""
    param_names = [
        r'$\alpha_a$', r'$\alpha_a^\prime$', r'$\alpha_r$', r'$\alpha_r^\prime$',
        r'$\beta_a$', r'$\beta_r$', r'$\delta_{ma}$', r'$\delta_{mr}$', r'$\gamma_a$',
        r'$\gamma_r$', r'$\gamma_c$', r'$\theta_a$', r'$\theta_r$', r'$\delta_a$', r'$\delta_r$'
    ]
    
    # Prior ranges for Vilar parameters
    prior_ranges = [
        (0, 100),      # alpha_a
        (0, 1000),     # alpha_a_prime
        (0, 10),       # alpha_r
        (0, 100),      # alpha_r_prime
        (0, 100),      # beta_a
        (0, 10),       # beta_r
        (0, 20),       # delta_ma
        (0, 4),        # delta_mr
        (0, 4),        # gamma_a
        (0, 2),        # gamma_r
        (0, 10),        # gamma_c
        (0, 10),      # theta_a
        (0, 10),      # theta_r
        (0, 100),        # delta_a
        (0, 200)       # delta_r
    ]

    # Create figure with 3x5 layout (3 rows, 5 columns) - optimized for paper
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(18, 10.8))
    
    # First create all plots to get densities
    for i in range(15):
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        
        # Remove extreme outliers (keep 99% of data)
        samples_i = posterior_samples[:, i]
        p_low = np.percentile(samples_i, 0.5)
        p_high = np.percentile(samples_i, 99.5)
        samples_filtered = samples_i[(samples_i >= p_low) & (samples_i <= p_high)]
        
        # Get KDE estimate - explicitly set x-axis
        kde = sns.kdeplot(
            x=samples_filtered,
            ax=ax,
            color='#1f77b4',  # Matplotlib default blue
            fill=True,
            alpha=0.6,
            bw_adjust=0.5
        )
        
        # Set x-axis limits to prior range first
        ax.set_xlim(prior_ranges[i])
        
        # True value line - always draw it, extend limits if needed
        true_val = true_theta[i]
        ax.axvline(true_val, color='#d62728', linestyle='--', linewidth=4)
        
        # Extend x-axis if true value is outside prior range
        current_xlim = ax.get_xlim()
        if true_val < current_xlim[0] or true_val > current_xlim[1]:
            new_min = min(current_xlim[0], true_val - 0.1 * abs(true_val))
            new_max = max(current_xlim[1], true_val + 0.1 * abs(true_val))
            ax.set_xlim(new_min, new_max)
        
        # Set y-axis limits
        ax.set_ylim(bottom=0)
        
        # Remove x-label and place parameter name above plot
        ax.set_xlabel('')
        ax.text(0.5, 1.08, param_names[i], fontsize=35, fontweight='bold',
                ha='center', va='bottom', transform=ax.transAxes)
        
        # Only add y-label 'Density' to first plot in each row
        if col == 0:
            # Less padding for 1st row, more for 2nd and 3rd rows
            if row == 0:
                ax.set_ylabel('Density', fontsize=30, labelpad=10)
            else:
                ax.set_ylabel('Density', fontsize=30, labelpad=25)
        else:
            ax.set_ylabel('')
        
        # Remove top and right spines and make visible spines thicker
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Add subtle grid
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Apply clean formatter to both axes
        ax.xaxis.set_major_formatter(FuncFormatter(clean_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(clean_formatter))
        
        # Just add true value to existing ticks - keep it simple
        current_xticks = list(ax.get_xticks())
        if true_val not in current_xticks:
            current_xticks.append(true_val)
            current_xticks.sort()
            ax.set_xticks(current_xticks)
        
        # Make tick labels larger
        ax.tick_params(axis='both', which='major', labelsize=18, width=1.5)
        
        # Move only small values (0.01, 0.2, 0.5, etc.) to the right to avoid overlap with 0
        xlim = ax.get_xlim()
        for idx, label in enumerate(ax.get_xticklabels()):
            try:
                val = float(label.get_text().replace('−', '-'))
                # Only shift small positive values that are close to 0
                if 0 < val < 1 and val < xlim[1] * 0.15:
                    # Extra shift for very small values like 0.01
                    if val <= 0.02:
                        # Use offset to shift label to the right
                        from matplotlib.transforms import ScaledTranslation
                        offset = ScaledTranslation(30/72., 0, ax.figure.dpi_scale_trans)
                        label.set_transform(label.get_transform() + offset)
                    else:
                        label.set_horizontalalignment('left')
            except:
                pass

    # Create a legend at the top of the figure with a box
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

    # Adjust layout to make room for legend at top
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space at top for legend
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close()


# def plot_posterior_predictive(posterior_samples, true_ts, save_path):
#     """Plot posterior predictive distributions with selected samples in a single plot."""
#     np.random.seed(42)  # For reproducibility
#     random_indices = np.random.choice(posterior_samples.shape[0], size=3, replace=False)
#     selected_samples = posterior_samples[random_indices, :]
    
#     # Initialize Vilar model and solver
#     vilar_model = Vilar_Oscillator()
#     solver = SSACSolver(model=vilar_model)
    
#     # Simulate time series data using selected samples
#     selected_data = np.array([
#         simulator(sample[np.newaxis, :], vilar_model, solver, transform=True)
#         for sample in selected_samples
#     ])
#     selected_data = selected_data.squeeze(1)
    
#     # Plot settings
#     time = np.arange(true_ts.shape[1])
#     features = ['C', 'A', 'R']
#     markers = ['o', 's', '^']  # Different markers for each species
#     colors = ['blue', 'red', 'green']  # Different colors for each species
    
#     # Create a single plot
#     plt.figure(figsize=(12, 8))
    
#     # Plot true observations for each species with different markers and colors
#     for i, (feature, marker, color) in enumerate(zip(features, markers, colors)):
#         plt.plot(time, true_ts[i], label=f'True {feature}', color=color,
#                  marker=marker, markevery=20, markersize=8, linewidth=2)
    
#     plt.xlabel('Time', fontsize=14)
#     plt.ylabel('Species Population', fontsize=14)
#     plt.title('Time Series of Species Populations', fontsize=16)
#     plt.legend(loc='upper right', fontsize=12)
#     plt.grid(True, alpha=0.3)
    
#     # Remove top and right spines
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
    
#     plt.tight_layout()
#     plt.savefig(save_path, format='pdf', bbox_inches='tight')
#     plt.close()


# def plot_posterior_predictive(posterior_samples, true_ts, save_path):
#     """Plot posterior predictive distributions with selected samples in a single plot."""
#     np.random.seed(42)
#     random_indices = np.random.choice(posterior_samples.shape[0], size=3, replace=False)
#     selected_samples = posterior_samples[random_indices, :]
    
#     # Initialize Vilar model and solver
#     vilar_model = Vilar_Oscillator()
#     solver = SSACSolver(model=vilar_model)
    
#     # Simulate time series data using selected samples
#     selected_data = np.array([
#         simulator(sample[np.newaxis, :], vilar_model, solver, transform=True)
#         for sample in selected_samples
#     ])
#     selected_data = selected_data.squeeze(1)   # expected shape: (3, 3, T)
    
#     time = np.arange(true_ts.shape[1])
#     features = ['C', 'A', 'R']
#     markers = ['o', 's', '^']
#     colors = ['blue', 'red', 'green']
    
#     plt.figure(figsize=(12, 8))
    
#     for i, (feature, marker, color) in enumerate(zip(features, markers, colors)):
#         # plot generated trajectories first
#         for j in range(selected_data.shape[0]):
#             plt.plot(
#                 time,
#                 selected_data[j, i, :],
#                 color=color,
#                 alpha=0.25,
#                 linewidth=1.5,
#                 label='Generated' if (i == 0 and j == 0) else None
#             )
        
#         # plot true trajectory on top
#         plt.plot(
#             time,
#             true_ts[i],
#             color=color,
#             marker=marker,
#             markevery=20,
#             markersize=6,
#             linewidth=2.5,
#             label=f'True {feature}'
#         )
    
#     plt.xlabel('Time', fontsize=14)
#     plt.ylabel('Species Population', fontsize=14)
#     plt.title('Time Series of Species Populations', fontsize=16)
#     plt.legend(loc='upper right', fontsize=12)
#     plt.grid(True, alpha=0.3)
    
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
    
#     plt.tight_layout()
#     plt.savefig(save_path, format='pdf', bbox_inches='tight')
#     plt.close()

def plot_posterior_predictive(posterior_samples, observed_data, save_path):
    """Plot posterior predictive checks comparing simulated vs observed data."""
    # Initialize model and solver
    model = Vilar_Oscillator()
    solver = SSACSolver(model=model)
    
    # Select random subset of posterior samples for simulation
    np.random.seed(42)
    num_samples = 3
    random_indices = np.random.choice(len(posterior_samples), num_samples, replace=False)
    selected_samples = posterior_samples[random_indices]
    
    # Simulate trajectories for selected samples
    simulated_data = []
    for params in selected_samples:
        sim_result = simulator(params[np.newaxis, :], model, solver, transform=True)
        sim_result = sim_result.squeeze(0)
        simulated_data.append(sim_result)
    simulated_data = np.array(simulated_data)
    
    # Plot settings
    species_names = ['Complex (C)', 'Activator (A)', 'Repressor (R)']
    colors = ['#2E86C1', '#E74C3C', '#2ECC71']
    time = np.arange(observed_data.shape[1])
    
    # Create figure with specific size for two-column paper
    fig, axes = plt.subplots(3, 1, figsize=(7, 8))
    plt.subplots_adjust(hspace=0.4)
    
    for i, (ax, species, color) in enumerate(zip(axes, species_names, colors)):
        # Plot observed data first
        true_line = ax.plot(
            time,
            observed_data[i],
            color=color,
            linewidth=1.5,
            label='True',
            zorder=5
        )[0]
        
        # Plot simulated trajectories
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
        
        # Axis labels
        ax.set_ylabel('Population', fontsize=12)
        if i == 2:
            ax.set_xlabel('Time', fontsize=12)
        
        # Species title
        ax.text(
            0.3, 1.05, species,
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            ha='center'
        )
        
        # Styling
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Legend
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
    """Generate plots for all simulation budgets and runs."""
    os.makedirs('vilar_plots', exist_ok=True)
    
    # Process each file in posterior_samples directory
    for file_name in os.listdir('posterior_samples'):
        if not file_name.endswith('.npz'):
            continue

        print(f"Processing {file_name}")
        file_path = os.path.join('posterior_samples', file_name)
        data = np.load(file_path, allow_pickle=True)
        
        # Extract budget directly from filename
        match = re.search(r'budget(\d+)', file_name)
        if match:
            budget = match.group(1)
        else:
            budget = "unknown"
        
        # Get posterior samples
        if 'theta_samples' in data:
            samples_key = 'theta_samples'
        elif 'posterior_samples' in data:
            samples_key = 'posterior_samples'
        else:
            print(f"Warning: No samples found in {file_name}. Available keys: {list(data.keys())}")
            continue
        
        posterior_samples = data[samples_key]
        
        # Get true parameters
        if 'true_theta' in data:
            true_params = data['true_theta']
        elif 'true_parameters' in data:
            true_params = data['true_parameters']
        else:
            print(f"Warning: No true parameters found in {file_name}. Available keys: {list(data.keys())}")
            continue
        
        # Generate plots with extremely simple filenames
        plot_base_name = f'vilar_plots/{budget}'
        
        plot_posterior_distributions(
            posterior_samples,
            true_params,
            f'{plot_base_name}_dist.pdf'
        )
        
        # Load true time series from dataset file, not posterior file
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


# def main():
#     """Generate plots for all simulation budgets and runs."""
#     os.makedirs('vilar_plots', exist_ok=True)
    
#     # Process each file in posterior_samples directory
#     for file_name in os.listdir('posterior_samples'):
#         if file_name.endswith('.npz'):
#             print(f"Processing {file_name}")
#             file_path = os.path.join('posterior_samples', file_name)
#             data = np.load(file_path, allow_pickle=True)
            
#             # Extract budget directly from filename
#             match = re.search(r'budget(\d+)', file_name)
#             if match:
#                 budget = match.group(1)
#             else:
#                 budget = "unknown"
            
#             # Check and fix denormalization if needed
#             # Try both key names (for backward compatibility)
#             if 'theta_samples' in data:
#                 samples_key = 'theta_samples'
#             elif 'posterior_samples' in data:
#                 samples_key = 'posterior_samples'
#             else:
#                 print(f"Warning: No samples found in {file_name}. Available keys: {list(data.keys())}")
#                 continue
                
#             # Get posterior samples (already properly denormalized during sampling)
#             posterior_samples = data[samples_key]
            
#             # Get true parameters (handle both key names)
#             true_params = data['true_theta'] if 'true_theta' in data else data['true_parameters']
            
#             # Generate plots with extremely simple filenames
#             plot_base_name = f'vilar_plots/{budget}'
            
#             plot_posterior_distributions(
#                 posterior_samples,
#                 true_params,
#                 f'{plot_base_name}_dist.pdf'
#             )
            
#             # Get time series data (handle both key names)
#             if 'true_ts' in data:
#                 ts_data = data['true_ts']
#             elif 'observed_data' in data:
#                 ts_data = data['observed_data']
#             else:
#                 print(f"Warning: No time series data found in {file_name}. Available keys: {list(data.keys())}")
#                 continue
                
#             plot_posterior_predictive(
#                 posterior_samples,
#                 ts_data,
#                 f'{plot_base_name}_pred.pdf'
#             )
            
#             print(f"Saved plots for budget {budget}")

if __name__ == "__main__":
    main()
