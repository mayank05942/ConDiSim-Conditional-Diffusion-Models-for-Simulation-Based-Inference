import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import numpy as np
import matplotlib.pyplot as plt
import os
#from bayesflow.diagnostics import plot_sbc_ecdf
from bayesflow.diagnostics import calibration_ecdf



def generate_sbc_plots(results_dir, output_dir, task_name, budgets, num_runs):
    """
    Generates and saves SBC ECDF plots from previously saved posterior draws and true parameters.

    Parameters:
    - results_dir: Directory containing the .npz files with posterior draws and true theta.
    - output_dir: Directory to save the resulting SBC ECDF plots.
    - task_name: Name of the task being processed.
    - budgets: List of simulation budgets to process.
    - num_runs: Number of runs for each budget.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over budgets and runs
    for budget in budgets:
        for run in range(1, num_runs + 1):
            # Construct file path for the .npz file
            file_path = os.path.join(results_dir, f"{task_name}_sbc_draws_budget_{budget}_run_{run}.npz")

            # Load the SBC results
            try:
                data = np.load(file_path)
                posterior_draws = data["posterior_draws"]
                theta_true = data["theta_true"]
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                continue

            # Plot ECDF for SBC diagnostics
            plt.figure(figsize=(8, 6))
            num_params = posterior_draws.shape[-1]
            parameter_names = [f"$\\theta_{{{i+1}}}$" for i in range(num_params)]

            fig = calibration_ecdf(
                posterior_draws,
                theta_true,
                ecdf_bands_kwargs=dict(confidence=0.90),
                difference=True,
                stacked=False,
                variable_names=parameter_names,
                label_fontsize=18,
                legend_fontsize=16,
                title_fontsize=20,
                tick_fontsize=14,
                rank_ecdf_color="#1f77b4",
                fill_color="grey",
            )
                        # fig = plot_sbc_ecdf(
            #     posterior_draws,
            #     theta_true,
            #     ecdf_bands_kwargs=dict(confidence=0.90),
            #     difference=True,
            #     stacked=False,
            #     label_fontsize=18,
            #     legend_fontsize=16,
            #     title_fontsize=20,
            #     tick_fontsize=14,
            #     rank_ecdf_color="#1f77b4",
            #     fill_color="grey",
            # )

            # Remove the legend from all axes
            for ax in fig.axes:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()  # Remove the legend

            # Save the plot
            plot_filename = f"{task_name}_ECDF_budget_{budget}_run_{run}.pdf"
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            print(f"Plot saved to {plot_path}")


def main():
    # Define task name and paths
    task_name = "lotka_volterra"
    results_dir = f"/cephyr/users/nautiyal/Alvis/diffusion/results_15aug/{task_name}"  # Path where SBC .npz files are stored
    output_dir = f"/cephyr/users/nautiyal/Alvis/diffusion/ECDF/{task_name}"  # Path to save the SBC ECDF plots

    # Define budgets and number of runs
    budgets = [30000]  # Define budgets used during the experiments
    num_runs = 5  # Number of runs for each budget

    # Generate and save SBC ECDF plots
    generate_sbc_plots(results_dir, output_dir, task_name, budgets, num_runs)

if __name__ == "__main__":
    main()


