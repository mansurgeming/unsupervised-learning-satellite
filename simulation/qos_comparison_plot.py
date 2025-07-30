import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def plot_qos_comparison(m_list, baseline_dir, model_dir, num_ut):
    """
    Computes and plots the average QoS satisfaction rate.

    Args:
        m_list (list): List of satellite counts to plot.
        baseline_dir (str): Path to the baseline results directory.
        model_dir (str): Path to the model results directory.
        num_ut (int): The fixed number of User Terminals.

    Returns:
        matplotlib.figure.Figure: The figure object of the generated plot.
    """
    def compute_stats_from_folder(folder_path):
        avg_list, std_list = [], []
        for M in m_list:
            file_path = os.path.join(folder_path, f"test_result_{M}sat.csv")
            if not os.path.exists(file_path):
                avg_list.append(np.nan)
                std_list.append(0)
                continue
            
            df = pd.read_csv(file_path)
            qos_fraction = df["qos_count"] / num_ut
            avg_list.append(qos_fraction.mean())
            std_list.append(qos_fraction.std())
        return avg_list, std_list

    # --- Calculate Data ---
    avg_model, std_model = compute_stats_from_folder(model_dir)
    avg_baseline, std_baseline = compute_stats_from_folder(baseline_dir)

    # --- Plotting Logic ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(m_list))
    width = 0.35

    ax.bar(x - width/2, avg_model, width, yerr=std_model, capsize=6,
           label="Model", color='salmon', edgecolor='black')
    ax.bar(x + width/2, avg_baseline, width, yerr=std_baseline, capsize=6,
           label="Baseline", color='skyblue', edgecolor='black')

    # Labels on bars
    for i, val in enumerate(avg_model):
        if not np.isnan(val):
            ax.text(x[i] - width/2, val + 0.03, f"{val:.2f}", ha='center')
    for i, val in enumerate(avg_baseline):
        if not np.isnan(val):
            ax.text(x[i] + width/2, val + 0.03, f"{val:.2f}", ha='center')

    # Layout and labels
    ax.set_xticks(x)
    ax.set_xticklabels(m_list)
    ax.set_xlabel("Number of Satellites (M)", fontsize=12)
    ax.set_ylabel("Average QoS Satisfaction Rate", fontsize=12)
    ax.set_title("QoS Satisfaction Rate vs. Satellite Count", fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()
    
    return fig

# --- Standalone Execution Block ---
if __name__ == '__main__':
    print("Running qos_comparison_plot.py as a standalone script...")

    try:
        # Add the project root to the path to find the config file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        sys.path.append(project_root)
        
        # Dynamically import settings from config.py
        from config import SATELLITE_LIST, UT_COUNT, RESULTS_MODEL_DIR, RESULTS_BASELINE_DIR
        
        M_LIST_TEST = SATELLITE_LIST
        UT_COUNT_TEST = UT_COUNT
        MODEL_DIR_TEST = RESULTS_MODEL_DIR
        BASELINE_DIR_TEST = RESULTS_BASELINE_DIR

    except ImportError:
        print("❌ Could not import from config.py. Using default fallback values for testing.")
        # Fallback configuration if config.py is not found
        M_LIST_TEST = [1, 6]
        UT_COUNT_TEST = 5
        MODEL_DIR_TEST = "results/model_results"
        BASELINE_DIR_TEST = "results/baseline_results"

    # --- Generate and Display the Plot ---
    fig = plot_qos_comparison(
        m_list=M_LIST_TEST,
        baseline_dir=BASELINE_DIR_TEST,
        model_dir=MODEL_DIR_TEST,
        num_ut=UT_COUNT_TEST
    )
    plt.show()