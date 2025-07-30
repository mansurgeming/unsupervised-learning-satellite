import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sys

def plot_avg_power_per_satellite(m_list, baseline_dir, model_dir):
    """
    Computes and plots the average transmit power per satellite.

    Args:
        m_list (list): List of satellite counts to plot.
        baseline_dir (str): Path to the baseline results directory.
        model_dir (str): Path to the model results directory.

    Returns:
        matplotlib.figure.Figure: The figure object of the generated plot.
    """
    def compute_avg_from_folder(folder_path):
        avg_per_satellite_data = []
        for M in m_list:
            file_path = os.path.join(folder_path, f"test_result_{M}sat.csv")
            if not os.path.exists(file_path):
                avg_per_satellite_data.append(np.nan)
                continue
            
            df = pd.read_csv(file_path)
            # Calculate the average of the total power for all samples
            avg_total_power = df["power_per_sat"].apply(lambda x: sum(json.loads(x))).mean()
            # Divide by the number of satellites (M) to get the average per satellite
            avg_per_satellite_data.append(avg_total_power / M if M > 0 else 0)
        return avg_per_satellite_data

    # --- Calculate Data ---
    avg_model = compute_avg_from_folder(model_dir)
    avg_baseline = compute_avg_from_folder(baseline_dir)

    # --- Plotting Logic ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(m_list))
    width = 0.35

    ax.bar(x - width/2, avg_model, width, label="Model", color='salmon', edgecolor='black')
    ax.bar(x + width/2, avg_baseline, width, label="Baseline", color='skyblue', edgecolor='black')

    # Trend lines
    ax.plot(x - width/2, avg_model, color='darkred', marker='o', linestyle='--')
    ax.plot(x + width/2, avg_baseline, color='steelblue', marker='s', linestyle='--')

    # Labels on bars
    max_val = max(np.nan_to_num(avg_model + avg_baseline))
    for i, val in enumerate(avg_model):
        if not np.isnan(val):
            ax.text(x[i] - width/2, val + 0.03 * max_val, f"{val:.1f}", ha='center')
    for i, val in enumerate(avg_baseline):
        if not np.isnan(val):
            ax.text(x[i] + width/2, val + 0.03 * max_val, f"{val:.1f}", ha='center')

    # Layout and labels
    ax.set_xticks(x)
    ax.set_xticklabels(m_list)
    ax.set_xlabel("Number of Satellites (M)", fontsize=12)
    ax.set_ylabel("Average Power per Satellite (Watt)", fontsize=12)
    ax.set_title("Comparison of Average Power per Satellite", fontsize=14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()
    
    return fig

# --- Standalone Execution Block ---
if __name__ == '__main__':
    print("Running avg_power_per_satellite_plot.py as a standalone script...")

    try:
        # Add the project root to the path to find the config file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        sys.path.append(project_root)
        
        # Dynamically import settings from config.py
        from config import SATELLITE_LIST, RESULTS_MODEL_DIR, RESULTS_BASELINE_DIR
        
        M_LIST_TEST = SATELLITE_LIST
        MODEL_DIR_TEST = RESULTS_MODEL_DIR
        BASELINE_DIR_TEST = RESULTS_BASELINE_DIR

    except ImportError:
        print("❌ Could not import from config.py. Using default fallback values for testing.")
        # Fallback configuration if config.py is not found
        M_LIST_TEST = [6, 10]
        MODEL_DIR_TEST = "results/model_results"
        BASELINE_DIR_TEST = "results/baseline_results"

    # --- Generate and Display the Plot ---
    fig = plot_avg_power_per_satellite(
        m_list=M_LIST_TEST,
        baseline_dir=BASELINE_DIR_TEST,
        model_dir=MODEL_DIR_TEST
    )
    plt.show()