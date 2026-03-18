import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D
import traceback

HHTask = None
jax = None
jnp = None

try:
    import jax
    import jax.numpy as jnp
    from scoresbibm.tasks.hhtask import HHTask
    PREDICTIVE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import JAX or HHTask: {e}")
    print("Posterior predictive plots will be disabled.")
    PREDICTIVE_AVAILABLE = False

def load_results(results_dir, budget, run_num):
    samples_path = os.path.join(results_dir, f'hh_posterior_samples_budget_{budget}_run_{run_num}.npz')
    
    if not os.path.exists(samples_path):
        raise ValueError(f"Results file not found: {samples_path}")
        
    print(f"Found results file: {samples_path}")
    
    data = np.load(samples_path, allow_pickle=True)
    
    posterior_samples = data['theta_samples']
    
    true_params = data.get('true_parameters', None)
    
    obs_data = {}
    voltage_trace = None
    for key in ['voltage_trace', 'true_V']:
        if key in data:
            voltage_trace = data[key]
            print(f"Found voltage trace with key: {key}")
            break
    
    # Check for energy trace with different possible key names
    energy_trace = None
    for key in ['energy_trace', 'true_H']:
        if key in data:
            energy_trace = data[key]
            print(f"Found energy trace with key: {key}")
            break
    
    # Create the observation data dictionary
    obs_data = {
        "V": voltage_trace,  # Voltage trace
        "H": energy_trace,  # Energy trace
        "summary_stats": None  # Not needed for plotting
    }
    
    return true_params, posterior_samples, obs_data

def plot_posterior_distributions(true_params, posterior_samples, save_path, budget, run_num):
    """Create KDE plots for posterior distributions of parameters."""
    posterior_samples = np.array(posterior_samples)
    # Clean invalid values in the first parameter
    posterior_samples[:, 0] = np.nan_to_num(posterior_samples[:, 0], nan=1.5, posinf=2.0, neginf=1.0)

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    axes = axes.flatten()

    param_names = [
        r'$C_m$ ($\mu$F/cm$^2$)',
        r'$g_{Na}$ (mS/cm$^2$)',
        r'$g_K$ (mS/cm$^2$)',
        r'$g_L$ (mS/cm$^2$)',
        r'$E_{Na}$ (mV)',
        r'$E_K$ (mV)',
        r'$E_V$ (mV)'
    ]

    prior_ranges = [
        (1.0, 2.0),          # Cm
        (60, 120),           # g_Na
        (10, 30),            # g_K
        (0.1, 0.5),          # g_L
        (40, 70),            # E_Na
        (-100, -60),         # E_K
        (-90, -60)           # E_V
    ]

    handles, labels = None, None

    for i in range(7):
        ax = axes[i]
        prior_min, prior_max = prior_ranges[i]
        
        # Get KDE estimate and normalize it
        kde = stats.gaussian_kde(posterior_samples[:, i])
        x_range = np.linspace(prior_min, prior_max, 200)
        density = kde(x_range)
        density = density / density.max()  # Normalize to [0,1]
        
        # Plot normalized density
        ax.fill_between(x_range, density, color='blue', alpha=0.6, 
                       label='Posterior')
        
        # True value line if available
        if true_params is not None:
            ax.axvline(true_params[i], color='red', linestyle='--', linewidth=2, label='True')
        
        # Prior range shading
        prior_min, prior_max = prior_ranges[i]
        ax.axvspan(prior_min, prior_max, color='gray', alpha=0.3, label='Prior Range')
        
        if i == 0:  # Special case for Cm
            ax.set_xlim(prior_min, prior_max)
        
        ax.set_title(param_names[i], fontsize=30, pad=10)
        
        # Show 'Density' label only on the first subplot of each column
        if i % 4 == 0:
            ax.set_ylabel("Density", fontsize=30, labelpad=8)
        else:
            ax.set_ylabel("")
        
        # Format numbers: remove .0 for whole numbers, keep one decimal for others
        def format_number(x):
            if abs(x - round(x)) < 1e-10:  # If it's effectively a whole number
                return f'{int(x)}'
            return f'{x:.1f}'
        
        # Special handling for ENa (index 4) to prevent text overlap
        if i == 4:  # ENa
            # Adjust the position of the maximum tick to prevent overlap
            original_ticks = [prior_min, prior_max]
            if true_params is not None:
                original_ticks.insert(1, true_params[i])
                adjusted_ticks = [prior_min, true_params[i], prior_max + 2]  # Move the 70 tick slightly right
                ax.set_xticks(adjusted_ticks)
                ax.set_xticklabels([format_number(prior_min), format_number(true_params[i]), format_number(prior_max)], 
                                 fontsize=16)
            else:
                adjusted_ticks = [prior_min, prior_max + 2]
                ax.set_xticks(adjusted_ticks)
                ax.set_xticklabels([format_number(prior_min), format_number(prior_max)], 
                                 fontsize=16)
            # Make sure the plot limits include our adjusted tick
            ax.set_xlim(prior_min - 1, prior_max + 3)
        else:
            if true_params is not None:
                ax.set_xticks([prior_min, true_params[i], prior_max])
                # Special handling for g_L to move small numbers left
                if i == 3:  # g_L
                    ax.set_xticklabels([format_number(prior_min), format_number(true_params[i]), format_number(prior_max)], 
                                      fontsize=16, ha='right')
                else:
                    ax.set_xticklabels([format_number(prior_min), format_number(true_params[i]), format_number(prior_max)], 
                                      fontsize=16)
            else:
                ax.set_xticks([prior_min, prior_max])
                if i == 3:  # g_L
                    ax.set_xticklabels([format_number(prior_min), format_number(prior_max)], 
                                      fontsize=16, ha='right')
                else:
                    ax.set_xticklabels([format_number(prior_min), format_number(prior_max)], 
                                      fontsize=16)
        
        # Set y-axis limits and ticks
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])
        # Remove original tick labels and add custom positioned ones
        ax.set_yticklabels(['', ''])
        # Add custom positioned y-axis labels
        ax.text(-0.05, 0.1, '0', transform=ax.get_yaxis_transform(),
                verticalalignment='center', horizontalalignment='right', fontsize=20)
        ax.text(-0.05, 1, '1', transform=ax.get_yaxis_transform(),
                verticalalignment='center', horizontalalignment='right', fontsize=20)
        
        # Thicker spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        if handles is None and labels is None:
            handles, labels = ax.get_legend_handles_labels()

    # Hide the extra subplot
    axes[-1].axis('off')

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.13), 
              fontsize=25, frameon=True, framealpha=1.0, edgecolor='black', ncol=3)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.92)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()

def plot_simple_traces(obs_data, save_path, budget, run_num):
    """Create simple plots for voltage and energy traces without requiring JAX."""
    # Check if we have the necessary data
    if obs_data["V"] is None or obs_data["H"] is None:
        print("Warning: Missing voltage or energy traces. Cannot create trace plots.")
        return
        
    V_obs = obs_data["V"]
    H_obs = obs_data["H"]
    
    # Create time axis
    t = np.linspace(0, 200, V_obs.shape[0])
    
    # Plot settings
    plt.rcParams.update({
        'font.size': 20,
        'axes.linewidth': 2.0,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'xtick.major.width': 2.0,
        'ytick.major.width': 2.0,
        'xtick.major.size': 10,
        'ytick.major.size': 10,
    })
    
    # Create voltage plot
    fig_v = plt.figure(figsize=(12, 6))
    ax_v = fig_v.add_subplot(111)
    
    # Plot observed voltage trace
    ax_v.plot(t, V_obs, color='darkblue', linewidth=2.5)
    
    # Customize voltage plot
    ax_v.set_xlabel("Time (ms)", labelpad=10)
    ax_v.set_ylabel("Voltage (mV)", labelpad=10)
    ax_v.set_title(f"Voltage Trace (Budget: {budget}, Run: {run_num})")
    
    # Remove top and right spines
    ax_v.spines['top'].set_visible(False)
    ax_v.spines['right'].set_visible(False)
    
    # Add grid
    ax_v.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # Set y-axis limits and ticks
    ax_v.set_ylim(-80, 65)
    ax_v.set_xlim(-5, 205)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save voltage plot
    voltage_save_path = os.path.join(os.path.dirname(save_path), f'hh_voltage_budget{budget}_run{run_num}.pdf')
    fig_v.savefig(voltage_save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig_v)
    
    # Create energy plot
    fig_e = plt.figure(figsize=(12, 6))
    ax_e = fig_e.add_subplot(111)
    
    # Plot observed energy trace
    ax_e.plot(t, H_obs, color='darkblue', linewidth=2.5)
    
    # Customize energy plot
    ax_e.set_xlabel("Time (ms)", labelpad=10)
    ax_e.set_ylabel("Energy", labelpad=10)
    ax_e.set_title(f"Energy Trace (Budget: {budget}, Run: {run_num})")
    
    # Remove top and right spines
    ax_e.spines['top'].set_visible(False)
    ax_e.spines['right'].set_visible(False)
    
    # Add grid
    ax_e.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # Set x-axis limits
    ax_e.set_xlim(-5, 205)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save energy plot
    energy_save_path = os.path.join(os.path.dirname(save_path), f'hh_energy_budget{budget}_run{run_num}.pdf')
    fig_e.savefig(energy_save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig_e)
    
    print(f"Created trace plots for budget {budget}, run {run_num}")


def plot_posterior_predictive(true_params, posterior_samples, obs_data, save_path, budget, run_num):
    """Create separate posterior predictive check plots for voltage and energy."""
    # Check if we have the necessary libraries
    if not PREDICTIVE_AVAILABLE:
        print("Cannot create posterior predictive plots: JAX or HHTask not available.")
        # Use the simple plotting function instead
        plot_simple_traces(obs_data, save_path, budget, run_num)
        return
        
    # Check if we have the necessary data
    if obs_data["V"] is None or obs_data["H"] is None:
        print("Warning: Missing voltage or energy traces. Cannot create posterior predictive plots.")
        return
        
    V_obs = obs_data["V"]
    H_obs = obs_data["H"]  # Energy trace
    
    try:
        # Initialize HH task and simulator
        hh_task = HHTask(backend="jax")
        rng_jax = jax.random.PRNGKey(420)
        simulator = hh_task.get_simulator()

        # Time axis
        t = jnp.linspace(0, 200, V_obs.shape[0])
        
        # Plot settings
        plt.rcParams.update({
            'font.size': 20,
            'axes.linewidth': 2.0,
            'axes.labelsize': 20,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'xtick.major.width': 2.0,
            'ytick.major.width': 2.0,
            'xtick.major.size': 10,
            'ytick.major.size': 10,
        })
        
        # Create voltage plot
        fig_v = plt.figure(figsize=(12, 6))
        ax_v = fig_v.add_subplot(111)
        
        # Create lines for legend first (thicker)
        legend_true = plt.Line2D([0], [0], color='darkblue', linewidth=12, label='True')
        legend_gen = plt.Line2D([0], [0], color='lightblue', linewidth=12, label='Generated')
        
        # Plot posterior predictive voltage traces
        posterior_samples = np.array(posterior_samples)
        rng_np = np.random.default_rng(seed=42)
        selected_samples = rng_np.choice(len(posterior_samples), size=10, replace=False)
        
        for idx in selected_samples:
            sample = posterior_samples[idx]
            V_sample, _, _ = simulator(rng_jax, jnp.array(sample))
            ax_v.plot(t, V_sample, color='lightblue', linewidth=1.5, 
                     marker='o', markersize=3, markevery=20, 
                     alpha=0.5, zorder=5)

        # Plot observed voltage trace on top
        ax_v.plot(t, V_obs, color='darkblue', linewidth=1.5, zorder=10)

        # Customize voltage plot
        ax_v.set_xlabel("Time (ms)", labelpad=10)
        ax_v.set_ylabel("Voltage (mV)", labelpad=10)
        
        # Add legend with box
        legend = fig_v.legend(handles=[legend_true, legend_gen],
                    loc='upper center', bbox_to_anchor=(0.5, 1.0),
                    ncol=2, fontsize=20, frameon=True,
                    framealpha=1.0, edgecolor='black')
        legend.get_frame().set_linewidth(1)
        
        # Remove top and right spines
        ax_v.spines['top'].set_visible(False)
        ax_v.spines['right'].set_visible(False)
        
        # Add grid
        ax_v.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Set y-axis limits and ticks
        ax_v.set_ylim(-80, 65)
        ax_v.set_xlim(-5, 205)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save voltage plot
        voltage_save_path = os.path.join(os.path.dirname(save_path), f'hh_voltage_budget{budget}_run{run_num}.pdf')
        fig_v.savefig(voltage_save_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig_v)

        # Create energy plot with similar formatting
        fig_e = plt.figure(figsize=(12, 6))
        ax_e = fig_e.add_subplot(111)
        
        # Create lines for legend first (thicker)
        legend_true = plt.Line2D([0], [0], color='darkblue', linewidth=12, label='True')
        legend_gen = plt.Line2D([0], [0], color='lightblue', linewidth=12, label='Generated')
        
        # Plot posterior predictive energy traces
        for idx in selected_samples:
            sample = posterior_samples[idx]
            _, H_sample, _ = simulator(rng_jax, jnp.array(sample))
            ax_e.plot(t, H_sample, color='lightblue', linewidth=1.5,
                     marker='o', markersize=3, markevery=20,
                     alpha=0.5, zorder=5)

        # Plot observed energy trace on top
        ax_e.plot(t, H_obs, color='darkblue', linewidth=1.5, zorder=10)

        # Customize energy plot
        ax_e.set_xlabel("Time (ms)", labelpad=10)
        ax_e.set_ylabel("Energy", labelpad=10)
        
        # Add legend with box
        legend = fig_e.legend(handles=[legend_true, legend_gen],
                    loc='upper center', bbox_to_anchor=(0.5, 1.1), 
                    ncol=2, fontsize=25, frameon=True,
                    framealpha=1.0, edgecolor='black')
        legend.get_frame().set_linewidth(1)
        
        # Remove top and right spines
        ax_e.spines['top'].set_visible(False)
        ax_e.spines['right'].set_visible(False)
        
        # Add grid
        ax_e.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Set x-axis limits
        ax_e.set_xlim(-5, 205)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save energy plot
        energy_save_path = os.path.join(os.path.dirname(save_path), f'hh_energy_budget{budget}_run{run_num}.pdf')
        fig_e.savefig(energy_save_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig_e)
    except Exception as e:
        print(f"Error creating posterior predictive plots: {str(e)}")
        print("This might be due to JAX/Symformer compatibility issues.")

def main():
    # Base results directory
    results_dir = "results/hh"
    plots_dir = "plots/hh"
    
    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    # Define the budgets and runs to process
    budgets = [10000, 20000, 30000]
    runs = [1]  # Assuming run 1 for all budgets
    
    # Process each simulation budget and run
    for budget in budgets:
        for run in runs:
            try:
                # Load results for this budget and run
                true_params, posterior_samples, obs_data = load_results(
                    results_dir, budget, run
                )
                
                print(f"\nLoaded results for budget {budget}, run {run}:")
                print(f"Posterior samples shape: {posterior_samples.shape}")
                if true_params is not None:
                    print(f"True parameters: {true_params}")
                else:
                    print("True parameters not available")
                
                # Check if voltage and energy traces are available
                if obs_data["V"] is not None:
                    print(f"Voltage trace shape: {obs_data['V'].shape}")
                else:
                    print("Voltage trace not available")
                    
                if obs_data["H"] is not None:
                    print(f"Energy trace shape: {obs_data['H'].shape}")
                else:
                    print("Energy trace not available")
                
                # Generate and save plots
                posterior_plot_path = os.path.join(
                    plots_dir, f"hh_posterior_budget{budget}_run{run}.pdf"
                )
                predictive_plot_path = os.path.join(
                    plots_dir, f"hh_voltage_energy_budget{budget}_run{run}.pdf"
                )
                
                # Plot posterior distributions
                plot_posterior_distributions(
                    true_params, posterior_samples, posterior_plot_path, budget, run
                )
                
                # Plot posterior predictive checks if we have the necessary data
                if obs_data["V"] is not None and obs_data["H"] is not None:
                    plot_posterior_predictive(
                        true_params, posterior_samples, obs_data, predictive_plot_path, budget, run
                    )
                else:
                    print("Skipping posterior predictive plots due to missing observation data")
                
                print(f"Generated plots for budget {budget}, run {run}")
                
            except Exception as e:
                print(f"Error processing budget {budget}, run {run}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

if __name__ == "__main__":
    main()
