import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker

# Path to the main results directory containing C2ST results
c2st_results_dir = '/cephyr/users/nautiyal/Alvis/diffusion/c2st_results'

# Path to save the plots
plots_dir = '/cephyr/users/nautiyal/Alvis/diffusion/plots/c2st_plots'

# Create the plots directory if it doesn't exist
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Function to load C2ST results
def load_c2st_results(task_name):
    """Load C2ST results for a given task."""
    task_results_file = os.path.join(c2st_results_dir, task_name, f'{task_name}_c2st_results.json')

    if not os.path.exists(task_results_file):
        print(f"No C2ST results found for task '{task_name}'.")
        return None

    with open(task_results_file, 'r') as f:
        c2st_results = json.load(f)

    return c2st_results

# Function to plot C2ST accuracy with standard deviation as error bars
def plot_c2st_accuracy(task_name, c2st_results):
    """Plot and save the mean C2ST accuracy versus simulation budget with error bars."""
    budgets_to_include = [10000, 20000, 30000]
    budgets = []
    mean_accuracies = []
    std_devs = []

    for budget_str, runs in c2st_results.items():
        budget = int(budget_str)
        if budget in budgets_to_include:
            accuracies = [run_data['c2st_accuracy'] for run_data in runs.values()]
            if accuracies:
                budgets.append(budget)
                mean_accuracies.append(np.mean(accuracies))
                std_devs.append(np.std(accuracies))  # Calculate standard deviation

    if not budgets:
        print(f"No data to plot for task '{task_name}'.")
        return

    # Sort the results by budget for better plotting
    sorted_indices = np.argsort(budgets)
    budgets = np.array(budgets)[sorted_indices]
    mean_accuracies = np.array(mean_accuracies)[sorted_indices]
    std_devs = np.array(std_devs)[sorted_indices]

    # Plotting with Seaborn and Matplotlib for aesthetics
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    plt.errorbar(budgets, mean_accuracies, yerr=std_devs, marker='o', linestyle='-', linewidth=2.5, markersize=10,
                 color='#4a90e2', capsize=5, label='Mean ± STD')

    # Set aesthetic properties
    plt.ylim(0.5, 1.0)
    plt.xlim(9000, 31000)
    plt.xticks(budgets, ['10k', '20k', '30k'], fontsize=12)
    plt.yticks(np.arange(0.5, 1.0, 0.1), fontsize=12)
    plt.xlabel('Simulation Budget', fontsize=14, labelpad=10)
    plt.ylabel('Mean C2ST Accuracy', fontsize=14, labelpad=10)
    #plt.title(task_name, fontsize=16, pad=15)
    plt.grid(visible=True, which='major', color='#d9d9d9', linestyle='--', linewidth=0.75)
    plt.grid(visible=True, which='minor', color='#f0f0f0', linestyle='--', linewidth=0.5)
    plt.gca().set_axisbelow(True)  # Grid lines behind plot elements

    # Remove the top and right spines for a cleaner look
    sns.despine()

    # Save the plot as a high-quality PDF for publication purposes
    plot_file_path = os.path.join(plots_dir, f'{task_name}_c2st_results.pdf')
    plt.tight_layout()
    plt.savefig(plot_file_path, format='pdf', bbox_inches='tight')
    print(f"Plot saved to {plot_file_path}")
    plt.close()

# Main function
def main(task_names):
    """Main function to load and plot C2ST results for specified tasks."""
    if isinstance(task_names, str):
        task_names = [task_names]

    for task_name in task_names:
        c2st_results = load_c2st_results(task_name)
        if c2st_results:
            plot_c2st_accuracy(task_name, c2st_results)


# Main function
def main(task_names):
    """Main function to load and plot C2ST results for specified tasks."""
    if isinstance(task_names, str):
        task_names = [task_names]

    for task_name in task_names:
        c2st_results = load_c2st_results(task_name)
        if c2st_results:
            plot_c2st_accuracy(task_name, c2st_results)
if __name__ == "__main__":

    task_input = 'all'
    #task_input = ['two_moons', 'sir', 'gaussian_mixture', 'bernoulli_glm', 'bernoulli_glm_raw']
    if task_input == 'all':
        task_names = [
            'bernoulli_glm',
            'bernoulli_glm_raw',
            'gaussian_linear',
            'gaussian_linear_uniform',
            'gaussian_mixture',
            'lotka_volterra',
            'sir',
            'slcp',
            'slcp_distractors',
            'two_moons'
        ]
    else:
        task_names = task_input if isinstance(task_input, list) else [task_input]

    main(task_names)
