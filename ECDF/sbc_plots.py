import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import numpy as np
import matplotlib.pyplot as plt
import os
from bayesflow.diagnostics import calibration_ecdf



def generate_sbc_plots(results_dir, output_dir, task_name, budgets, num_runs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for budget in budgets:
        for run in range(1, num_runs + 1):
            file_path = os.path.join(results_dir, f"{task_name}_sbc_draws_budget_{budget}_run_{run}.npz")

            try:
                data = np.load(file_path)
                posterior_draws = data["posterior_draws"]
                theta_true = data["theta_true"]
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                continue

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

            for ax in fig.axes:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()

            plot_filename = f"{task_name}_ECDF_budget_{budget}_run_{run}.pdf"
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            print(f"Plot saved to {plot_path}")


def main():
    task_name = "lotka_volterra"
    results_dir = f"results/{task_name}"
    output_dir = f"ECDF/{task_name}"

    budgets = [30000]
    num_runs = 5

    generate_sbc_plots(results_dir, output_dir, task_name, budgets, num_runs)

if __name__ == "__main__":
    main()


