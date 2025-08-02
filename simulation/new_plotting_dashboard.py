import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import matplotlib
import streamlit as st
import math

# Pastikan backend matplotlib diatur sebelum pemanggilan plot apa pun
matplotlib.use('Agg')

# Asumsikan file ini ada dan berisi variabel yang diperlukan
from config import SATELLITE_LIST, R_MIN, UT_COUNT

# ===================================================================
# Bagian ini berisi semua fungsi plot Anda yang sudah benar.
# Tidak ada perubahan logika di sini.
# ===================================================================

def plot_cdf(model_dir, baseline_dir, selected_satellites):
    if not selected_satellites:
        st.warning("Please select at least one satellite configuration to display the CDF plot.")
        return

    num_plots = len(selected_satellites)
    cols = 2 if num_plots > 1 else 1
    rows = math.ceil(num_plots / cols)
    
    fig, axs = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), squeeze=False)
    axs = axs.flatten()

    for idx, M in enumerate(selected_satellites):
        baseline_path = os.path.join(baseline_dir, f"test_result_{M}sat.csv")
        model_path = os.path.join(model_dir, f"test_result_{M}sat.csv")

        if not os.path.exists(baseline_path) or not os.path.exists(model_path):
            axs[idx].text(0.5, 0.5, f"Data for {M} satellites not found", ha='center', va='center')
            axs[idx].set_title(f"Satellites = {M}")
            continue

        def load_rates(path):
            df = pd.read_csv(path)
            all_rates = [rate for r_str in df['rate_per_ut'] for rate in json.loads(r_str)]
            all_rates_sorted = np.sort(all_rates)
            cdf = np.arange(1, len(all_rates_sorted) + 1) / len(all_rates_sorted)
            return all_rates_sorted, cdf

        x_base, y_base = load_rates(baseline_path)
        x_model, y_model = load_rates(model_path)

        axs[idx].plot(x_base, y_base, label="Baseline", marker='.', linestyle='-')
        axs[idx].plot(x_model, y_model, label="Model", marker='.', linestyle='-')
        axs[idx].axvline(x=R_MIN, color='red', linestyle='--', linewidth=2, label=f"R_min = {R_MIN}")
        axs[idx].set_title(f"Satellites = {M}")
        axs[idx].set_xlabel("Rate per UT (bps)")
        axs[idx].set_ylabel("CDF")
        axs[idx].grid(True)
        axs[idx].legend()

    for idx in range(num_plots, len(axs)):
        axs[idx].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)


def plot_bar_chart(title, ylabel, compute_func, model_dir, baseline_dir, selected_satellites):
    if not selected_satellites:
        # Peringatan ini bisa di-skip jika sudah ada pengecekan di level atas
        return

    avg_model, std_model = compute_func(model_dir, selected_satellites)
    avg_baseline, std_baseline = compute_func(baseline_dir, selected_satellites)

    x = np.arange(len(selected_satellites))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width/2, avg_model, width, yerr=std_model, capsize=5, label="Model", color='salmon', edgecolor='black')
    ax.bar(x + width/2, avg_baseline, width, yerr=std_baseline, capsize=5, label="Baseline", color='skyblue', edgecolor='black')
    ax.plot(x - width/2, avg_model, color='darkred', marker='o', linestyle='--')
    ax.plot(x + width/2, avg_baseline, color='steelblue', marker='s', linestyle='--')

    ax.set_xticks(x)
    ax.set_xticklabels(selected_satellites)
    ax.set_xlabel("Number of Satellites (M)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend()
    
    st.pyplot(fig)

def compute_avg_total_power(folder_path, selected_satellites):
    avg_power = [np.mean([sum(json.loads(p_str)) for p_str in pd.read_csv(os.path.join(folder_path, f"test_result_{M}sat.csv"))["power_per_sat"]]) if os.path.exists(os.path.join(folder_path, f"test_result_{M}sat.csv")) else np.nan for M in selected_satellites]
    return avg_power, None

def compute_avg_power_per_sat(folder_path, selected_satellites):
    avg_per_sat = [np.mean([sum(json.loads(p_str)) / M for p_str in pd.read_csv(os.path.join(folder_path, f"test_result_{M}sat.csv"))["power_per_sat"]]) if os.path.exists(os.path.join(folder_path, f"test_result_{M}sat.csv")) else np.nan for M in selected_satellites]
    return avg_per_sat, None

def compute_qos_stats(folder_path, selected_satellites):
    avg_list, std_list = [], []
    for M in selected_satellites:
        file_path = os.path.join(folder_path, f"test_result_{M}sat.csv")
        if os.path.exists(file_path):
            qos_fraction = pd.read_csv(file_path)["qos_count"] / UT_COUNT
            avg_list.append(qos_fraction.mean())
            std_list.append(qos_fraction.std())
        else:
            avg_list.append(np.nan)
            std_list.append(np.nan)
    return avg_list, std_list

def compute_avg_throughput(folder_path, selected_satellites):
    avg_throughput = [np.mean([sum(json.loads(r_str)) for r_str in pd.read_csv(os.path.join(folder_path, f"test_result_{M}sat.csv"))["rate_per_ut"]]) if os.path.exists(os.path.join(folder_path, f"test_result_{M}sat.csv")) else np.nan for M in selected_satellites]
    return avg_throughput, None

def plot_5_users_from_best_sample(satellite_count, best_sample_row):
    USER_COUNT = UT_COUNT
    sample_id = best_sample_row.name
    
    rates_for_5_users = json.loads(best_sample_row['rate_per_ut'])

    np.random.seed(42)
    x_sats = np.random.uniform(1, 10, size=satellite_count)
    y_sats = np.random.uniform(1, 10, size=satellite_count)
    x_users = np.random.uniform(1, 10, size=USER_COUNT)
    y_users = np.random.uniform(1, 10, size=USER_COUNT)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x_sats, y_sats, color='blue', marker='^', s=150, label="Satelit", edgecolors='black', zorder=5)

    for i in range(USER_COUNT):
        user_rate = rates_for_5_users[i]
        if user_rate >= R_MIN:
            ax.scatter(x_users[i], y_users[i], color='green', marker='o', s=100, edgecolors='black', zorder=10)
            for j in range(satellite_count):
                ax.plot([x_users[i], x_sats[j]], [y_users[i], y_sats[j]], color='green', linewidth=1.5)
        else:
            ax.scatter(x_users[i], y_users[i], color='red', marker='o', s=100, edgecolors='black', zorder=10)

    ax.scatter([], [], color='green', marker='o', s=100, edgecolors='black', label='Pengguna (QoS Terpenuhi)')
    ax.scatter([], [], color='red', marker='o', s=100, edgecolors='black', label='Pengguna (QoS Tdk Terpenuhi)')

    ax.set_title(f"Visualisasi Sampel Terbaik untuk {satellite_count} Satelit (ID: {sample_id})", fontsize=16)
    ax.set_xlabel("Posisi X")
    ax.set_ylabel("Posisi Y")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)
    
    st.pyplot(fig)


# ================================
# Fungsi utama untuk menjalankan semua plot
# ================================
def main(model_results_dir,
    baseline_results_dir,
    raw_data_dir,
    R_MIN):
    st.set_page_config(layout="wide")
    st.title("🛰️ Satellite Network Performance Dashboard")

    st.sidebar.header("Direktori Data")
    model_results_dir = st.sidebar.text_input("Model Results Directory", "results/model_results")
    baseline_results_dir = st.sidebar.text_input("Baseline Results Directory", "results/baseline_results")

    st.sidebar.header("Filter Konfigurasi")
    selected_satellites = st.sidebar.multiselect(
        "Pilih Konfigurasi Satelit untuk Ditampilkan:",
        options=SATELLITE_LIST,
        default=SATELLITE_LIST
    )
    selected_satellites.sort()

    if not selected_satellites:
        st.info("Silakan pilih minimal satu konfigurasi satelit dari sidebar untuk menampilkan data.")
        return # Hentikan eksekusi jika tidak ada yang dipilih

    # --- Tampilkan Plot Perbandingan ---
    st.header("Perbandingan Kinerja Model vs. Baseline")
    
    st.subheader("1. Cumulative Distribution Function (CDF) of User Rate")
    plot_cdf(model_results_dir, baseline_results_dir, selected_satellites)
    st.markdown("---")

    st.subheader("2. Analisis Penggunaan Daya dan Throughput")
    col1, col2 = st.columns(2)
    with col1:
        plot_bar_chart("Total Power Usage vs. Satellite Count", "Average Total Power (Watt)", compute_avg_total_power, model_results_dir, baseline_results_dir, selected_satellites)
    with col2:
        plot_bar_chart("Average Power per Satellite vs. Satellite Count", "Average Power per Satellite (Watt)", compute_avg_power_per_sat, model_results_dir, baseline_results_dir, selected_satellites)

    col3, col4 = st.columns(2)
    with col3:
         plot_bar_chart("QoS Satisfaction Rate vs. Satellite Count", "Average QoS Satisfaction Rate", compute_qos_stats, model_results_dir, baseline_results_dir, selected_satellites)
    with col4:
        plot_bar_chart("Total Network Throughput vs. Satellite Count", "Average Total Throughput (bps)", compute_avg_throughput, model_results_dir, baseline_results_dir, selected_satellites)
    st.markdown("---")


    # ===================================================================
    # BAGIAN BARU: Menampilkan Sampel Terbaik untuk SETIAP Konfigurasi
    # ===================================================================
    st.header("Visualisasi Sampel Terbaik per Konfigurasi")
    st.markdown("Buka setiap bagian di bawah untuk melihat visualisasi sampel dengan `qos_count` tertinggi untuk konfigurasi satelit tersebut.")

    for satellite_count in selected_satellites:
        # Gunakan st.expander untuk membuat bagian yang bisa di-collapse
        with st.expander(f"Lihat Sampel Terbaik untuk {satellite_count} Satelit"):
            best_sample_row = None
            best_sample_qos_count = -1
            best_sample_rate_total = -1.0
            
            try:
                file_path = os.path.join(model_results_dir, f"test_result_{satellite_count}sat.csv")
                if os.path.exists(file_path):
                    sample_data = pd.read_csv(file_path)
                    
                    # Cari baris terbaik HANYA di dalam file ini
                    for index, row in sample_data.iterrows():
                        if row['qos_count'] > best_sample_qos_count or \
                           (row['qos_count'] == best_sample_qos_count and row['rate_total'] > best_sample_rate_total):
                            best_sample_qos_count = row['qos_count']
                            best_sample_rate_total = row['rate_total']
                            best_sample_row = row
                    
                    # Jika baris terbaik ditemukan, tampilkan info dan plotnya
                    if best_sample_row is not None:
                        st.info(f"Sampel terbaik untuk **{satellite_count} satelit** ditemukan di index **{best_sample_row.name}**.\n\n"
                                f"- **QoS Count:** {best_sample_qos_count}\n\n"
                                f"- **Rate Total:** {best_sample_rate_total:.2f}")
                        
                        plot_5_users_from_best_sample(satellite_count, best_sample_row)
                    else:
                        st.warning("Tidak ada data valid yang ditemukan di file ini.")
                        
                else:
                    st.error(f"File tidak ditemukan: {file_path}")

            except Exception as e:
                st.error(f"Error saat memproses file untuk {satellite_count} satelit: {e}")


if __name__ == "__main__":
    main()