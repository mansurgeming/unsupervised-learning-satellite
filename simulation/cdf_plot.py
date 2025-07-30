import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

def plot_cdf(m_list, baseline_dir, model_dir, r_min):
    """
    Membuat dan menampilkan plot CDF untuk perbandingan model dan baseline.
    """
    num_plots = len(m_list)
    num_cols = min(num_plots, 2)
    num_rows = (num_plots + num_cols - 1) // num_cols
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 5 * num_rows), squeeze=False)
    axs = axs.flatten()

    for idx, M in enumerate(m_list):
        ax = axs[idx]
        baseline_path = os.path.join(baseline_dir, f"test_result_{M}sat.csv")
        model_path = os.path.join(model_dir, f"test_result_{M}sat.csv")

        if not os.path.exists(baseline_path) or not os.path.exists(model_path):
            print(f"Melewatkan M={M}: File hasil tidak ditemukan.")
            ax.text(0.5, 0.5, f"Data untuk M={M}\ntidak ditemukan", ha='center', va='center')
            ax.set_title(f"Satellites = {M}")
            continue

        def load_rates(path):
            df = pd.read_csv(path)
            all_rates = [rate for r_str in df['rate_per_ut'] for rate in json.loads(r_str)]
            all_rates_sorted = np.sort(all_rates)
            cdf = np.arange(1, len(all_rates_sorted) + 1) / len(all_rates_sorted)
            return all_rates_sorted, cdf

        x_base, y_base = load_rates(baseline_path)
        x_model, y_model = load_rates(model_path)

        ax.plot(x_base, y_base, label="Baseline", linestyle='--')
        ax.plot(x_model, y_model, label="Model")
        ax.axvline(x=r_min, color='red', linestyle=':', linewidth=1.5, label=f"R_min={r_min}")
        ax.set_title(f"Satellites = {M}")
        ax.set_xlabel("Rate per UT (bps/Hz)")
        ax.set_ylabel("CDF")
        ax.grid(True, alpha=0.5)
        ax.legend()

    for i in range(len(m_list), len(axs)):
        axs[i].set_visible(False)
        
    plt.tight_layout()
    return fig

# --- Blok Eksekusi Mandiri ---
if __name__ == '__main__':
    print("Menjalankan cdf_plot.py sebagai skrip mandiri untuk menampilkan plot...")
    
    # ==========================================================
    # === PERUBAHAN UTAMA ADA DI SINI ===
    # ==========================================================
    try:
        # Menambahkan path root proyek agar bisa mengimpor config.py
        # Ini mengasumsikan cdf_plot.py berada di dalam folder 'simulation'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        sys.path.append(project_root)
        
        # Impor konfigurasi secara dinamis
        from config import SATELLITE_LIST, R_MIN_THRESHOLD, RESULTS_MODEL_DIR, RESULTS_BASELINE_DIR
        
        M_LIST_TEST = SATELLITE_LIST
        R_MIN_TEST = R_MIN_THRESHOLD
        MODEL_DIR_TEST = RESULTS_MODEL_DIR
        BASELINE_DIR_TEST = RESULTS_BASELINE_DIR

    except ImportError:
        print("❌ Gagal mengimpor dari config.py. Menggunakan nilai default untuk pengujian.")
        # Konfigurasi fallback jika config.py tidak ditemukan
        M_LIST_TEST = [1, 6]
        R_MIN_TEST = 0.5
        MODEL_DIR_TEST = "results/model_results"
        BASELINE_DIR_TEST = "results/baseline_results"

    # Panggil fungsi untuk membuat plot
    fig = plot_cdf(M_LIST_TEST, BASELINE_DIR_TEST, MODEL_DIR_TEST, R_MIN_TEST)
    
    # Tampilkan plot seperti di Colab
    plt.show()