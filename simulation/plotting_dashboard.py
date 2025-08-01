import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import matplotlib
import streamlit as st
matplotlib.use('Agg')  # Ganti dengan Qt5Agg jika TkAgg tidak berhasil

from config import SATELLITE_LIST, R_MIN, UT_COUNT


# ================================
# Plot CDF per M dalam format 3x2
# ================================
def plot_cdf(model_dir, baseline_dir):
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    axs = axs.flatten()

    for idx, M in enumerate(SATELLITE_LIST):
        baseline_path = os.path.join(baseline_dir, f"test_result_{M}sat.csv")
        model_path = os.path.join(model_dir, f"test_result_{M}sat.csv")

        if not os.path.exists(baseline_path) or not os.path.exists(model_path):
            continue

        def load_rates(path):
            df = pd.read_csv(path)
            all_rates = []
            for r_str in df['rate_per_ut']:
                rates = json.loads(r_str)
                all_rates.extend(rates)
            all_rates = np.array(all_rates)
            all_rates_sorted = np.sort(all_rates)
            cdf = np.arange(1, len(all_rates_sorted)+1) / len(all_rates_sorted)
            return all_rates_sorted, cdf

        # Load data
        x_base, y_base = load_rates(baseline_path)
        x_model, y_model = load_rates(model_path)

        # Plot
        axs[idx].plot(x_base, y_base, label="Baseline")
        axs[idx].plot(x_model, y_model, label="Model")
        axs[idx].axvline(x=R_MIN, color='red', linestyle='--', linewidth=1, label="R_min" if idx == 0 else "")
        axs[idx].set_title(f"Satellites = {M}")
        axs[idx].set_xlabel("Rate per UT (bps)")
        axs[idx].set_ylabel("CDF")
        axs[idx].grid(True)
        axs[idx].legend()

    plt.tight_layout()
    # Tampilkan plot dengan Streamlit
    st.pyplot(fig)


# ================================
# AVG TOTAL TRANSMIT POWER VS NUMBER OF SATELLITES
# ================================
def plot_avg_total_power(model_dir, baseline_dir):
    def compute_avg_total_power(folder_path):
        avg_power = []
        for M in SATELLITE_LIST:
            file_path = os.path.join(folder_path, f"test_result_{M}sat.csv")
            if not os.path.exists(file_path):
                avg_power.append(np.nan)
                continue
            df = pd.read_csv(file_path)
            total_power_samples = []
            for p_str in df["power_per_sat"]:
                power_list = json.loads(p_str) if isinstance(p_str, str) else []
                total_power = sum(power_list)
                total_power_samples.append(total_power)
            avg_power.append(np.mean(total_power_samples))
        return avg_power

    avg_power_model = compute_avg_total_power(model_dir)
    avg_power_baseline = compute_avg_total_power(baseline_dir)

    x = np.arange(len(SATELLITE_LIST))
    width = 0.35

    # Membuat objek 'fig' untuk plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, avg_power_model, width, label="Model", color='salmon', edgecolor='black')
    bars2 = ax.bar(x + width/2, avg_power_baseline, width, label="Baseline", color='skyblue', edgecolor='black')

    ax.plot(x - width/2, avg_power_model, color='darkred', marker='o', linestyle='--')
    ax.plot(x + width/2, avg_power_baseline, color='steelblue', marker='s', linestyle='--')

    for i in range(len(SATELLITE_LIST)):
        if not np.isnan(avg_power_model[i]):
            ax.text(x[i] - width/2, avg_power_model[i] + 0.03 * max(avg_power_model), f"{avg_power_model[i]:.1f}", ha='center')
        if not np.isnan(avg_power_baseline[i]):
            ax.text(x[i] + width/2, avg_power_baseline[i] + 0.03 * max(avg_power_baseline), f"{avg_power_baseline[i]:.1f}", ha='center')

    ax.set_xticks(x)
    ax.set_xticklabels(SATELLITE_LIST)
    ax.set_xlabel("Number of Satellites (M)", fontsize=12)
    ax.set_ylabel("Average Total Power Usage (Watt)", fontsize=12)
    ax.set_title("Comparison of Total Power Usage vs. Satellite Count", fontsize=14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend()
    
    # Tampilkan plot dengan Streamlit
    st.pyplot(fig)


# ================================
# AVG TOTAL TRANSMIT POWER PER SATELLITE VS NUMBER OF SATELLITES
# ================================
def plot_avg_power_per_sat(model_dir, baseline_dir):
    def compute_avg_power_per_sat(folder_path):
        avg_per_sat = []
        for M in SATELLITE_LIST:
            file_path = os.path.join(folder_path, f"test_result_{M}sat.csv")
            if not os.path.exists(file_path):
                avg_per_sat.append(np.nan)
                continue

            df = pd.read_csv(file_path)
            total_power_list = df["power_per_sat"].apply(lambda x: sum(json.loads(x))).values
            avg_total = np.mean(total_power_list)
            avg_per_sat.append(avg_total / M)
        return avg_per_sat

    avg_model = compute_avg_power_per_sat(model_dir)
    avg_baseline = compute_avg_power_per_sat(baseline_dir)

    x = np.arange(len(SATELLITE_LIST))
    width = 0.35

    # Membuat objek 'fig' untuk plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Inisialisasi 'fig' di sini

    bars1 = ax.bar(x - width/2, avg_model, width, label="Model", color='salmon', edgecolor='black')
    bars2 = ax.bar(x + width/2, avg_baseline, width, label="Baseline", color='skyblue', edgecolor='black')

    ax.plot(x - width/2, avg_model, color='darkred', marker='o', linestyle='--')
    ax.plot(x + width/2, avg_baseline, color='steelblue', marker='s', linestyle='--')

    for i in range(len(SATELLITE_LIST)):
        if not np.isnan(avg_model[i]):
            ax.text(x[i] - width/2, avg_model[i] + 0.03 * max(avg_model), f"{avg_model[i]:.1f}", ha='center')
        if not np.isnan(avg_baseline[i]):
            ax.text(x[i] + width/2, avg_baseline[i] + 0.03 * max(avg_baseline), f"{avg_baseline[i]:.1f}", ha='center')

    ax.set_xticks(x)
    ax.set_xticklabels(SATELLITE_LIST)
    ax.set_xlabel("Number of Satellites (M)", fontsize=12)
    ax.set_ylabel("Average Power per Satellite (Watt)", fontsize=12)
    ax.set_title("Comparison of Average Power per Satellite vs. Satellite Count", fontsize=14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend()

    # Tampilkan plot dengan Streamlit
    st.pyplot(fig)


# ================================
# QOS COMPARISON
# ================================
def plot_qos_comparison(model_dir, baseline_dir):
    def compute_qos_stats(folder_path):
        avg_list = []
        std_list = []
        for M in SATELLITE_LIST:
            file_path = os.path.join(folder_path, f"test_result_{M}sat.csv")
            if not os.path.exists(file_path):
                avg_list.append(np.nan)
                std_list.append(0)
                continue
            df = pd.read_csv(file_path)
            qos_fraction = df["qos_count"] / 5
            avg_list.append(qos_fraction.mean())
            std_list.append(qos_fraction.std())
        return avg_list, std_list

    avg_model, std_model = compute_qos_stats(model_dir)
    avg_baseline, std_baseline = compute_qos_stats(baseline_dir)

    x = np.arange(len(SATELLITE_LIST))
    width = 0.35

    # Membuat objek 'fig' untuk plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Inisialisasi 'fig' di sini

    bars1 = ax.bar(x - width/2, avg_model, width, yerr=std_model, capsize=6,
                    label="Model", color='salmon', edgecolor='black')
    bars2 = ax.bar(x + width/2, avg_baseline, width, yerr=std_baseline, capsize=6,
                    label="Baseline", color='skyblue', edgecolor='black')

    for i in range(len(SATELLITE_LIST)):
        if not np.isnan(avg_model[i]):
            ax.text(x[i] - width/2, avg_model[i] + 0.03, f"{avg_model[i]:.2f}", ha='center')
        if not np.isnan(avg_baseline[i]):
            ax.text(x[i] + width/2, avg_baseline[i] + 0.03, f"{avg_baseline[i]:.2f}", ha='center')

    ax.set_xticks(x)
    ax.set_xticklabels(SATELLITE_LIST)
    ax.set_xlabel("Number of Satellites (M)", fontsize=12)
    ax.set_ylabel("Average QoS Satisfaction Rate", fontsize=12)
    ax.set_title("QoS Satisfaction Rate vs. Satellite Count", fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend()

    # Tampilkan plot dengan Streamlit
    st.pyplot(fig)


# ================================
# TOTAL NETWORK THROUGHPUT
# ================================
def plot_total_network_throughput(model_dir, baseline_dir):
    def compute_avg_throughput(folder_path):
        avg_throughput = []
        for M in SATELLITE_LIST:
            file_path = os.path.join(folder_path, f"test_result_{M}sat.csv")
            if not os.path.exists(file_path):
                avg_throughput.append(np.nan)
                continue

            df = pd.read_csv(file_path)
            total_per_sample = []
            for r_str, i_str in zip(df["rate_per_ut"], df["qos_count"]):
                rates = json.loads(r_str) if isinstance(r_str, str) else []
                total = sum(rates[:int(i_str)]) if isinstance(rates, list) else 0
                total_per_sample.append(total)
            avg_throughput.append(np.mean(total_per_sample))
        return avg_throughput

    avg_model = compute_avg_throughput(model_dir)
    avg_baseline = compute_avg_throughput(baseline_dir)

    x = np.arange(len(SATELLITE_LIST))
    width = 0.35

    # Membuat objek 'fig' untuk plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Inisialisasi 'fig' di sini

    bars1 = ax.bar(x - width/2, avg_model, width, label="Model", color='salmon', edgecolor='black')
    bars2 = ax.bar(x + width/2, avg_baseline, width, label="Baseline", color='skyblue', edgecolor='black')

    ax.plot(x - width/2, avg_model, color='darkred', marker='o', linestyle='--')
    ax.plot(x + width/2, avg_baseline, color='steelblue', marker='s', linestyle='--')

    for i in range(len(SATELLITE_LIST)):
        if not np.isnan(avg_model[i]):
            ax.text(x[i] - width/2, avg_model[i] + 0.03 * max(avg_model), f"{avg_model[i]:.2f}", ha='center')
        if not np.isnan(avg_baseline[i]):
            ax.text(x[i] + width/2, avg_baseline[i] + 0.03 * max(avg_baseline), f"{avg_baseline[i]:.2f}", ha='center')

    ax.set_xticks(x)
    ax.set_xticklabels(SATELLITE_LIST)
    ax.set_xlabel("Number of Satellites (M)", fontsize=12)
    ax.set_ylabel("Average Total Throughput (bps)", fontsize=12)
    ax.set_title("Comparison of Total Network Throughput vs. Satellite Count", fontsize=14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend()

    # Tampilkan plot dengan Streamlit
    st.pyplot(fig)


def plot_5_users_from_best_sample(satellite_count, best_sample_row, R_MIN):
    """
    Memvisualisasikan 5 pengguna dari satu sample_id terbaik.
    Warna garis ditentukan oleh status QoS setiap pengguna.

    :param satellite_count: Jumlah satelit dari konfigurasi tempat sampel terbaik ditemukan.
    :param best_sample_row: Satu baris Series pandas yang berisi data sampel terbaik.
    :param R_MIN: Nilai ambang batas rate.
    """
    USER_COUNT = 5  # Jumlah pengguna tetap 5
    sample_id = best_sample_row.name
    
    print(f"\n🎨 Memulai visualisasi untuk 5 pengguna dari sample_id terbaik: {sample_id}")
    print(f"   Pada konfigurasi {satellite_count} satelit.")

    # Ambil array rate untuk 5 pengguna dari baris terbaik
    rates_for_5_users = json.loads(best_sample_row['rate_per_ut'])

    # Tentukan posisi acak yang konsisten
    np.random.seed(42)
    x_sats = np.random.uniform(1, 10, size=satellite_count)
    y_sats = np.random.uniform(1, 10, size=satellite_count)
    x_users = np.random.uniform(1, 10, size=USER_COUNT)
    y_users = np.random.uniform(1, 10, size=USER_COUNT)

    # Membuat plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x_users, y_users, color='red', marker='o', s=100, label="Pengguna", edgecolors='black', zorder=10)
    ax.scatter(x_sats, y_sats, color='blue', marker='^', s=150, label="Satelit", edgecolors='black', zorder=5)

    # Loop untuk setiap 5 PENGGUNA
    for i in range(USER_COUNT):
        user_rate = rates_for_5_users[i]
        
        # Tentukan warna untuk PENGGUNA ini berdasarkan rate-nya
        if user_rate >= R_MIN:
            line_color = 'green'
        else:
            line_color = 'yellow'
            
        # Gambar garis dari pengguna ini ke SEMUA satelit dengan warna yang sudah ditentukan
        for j in range(satellite_count):
            ax.plot([x_users[i], x_sats[j]],
                    [y_users[i], y_sats[j]],
                    color=line_color,
                    linewidth=1.5)

    # Pengaturan detail plot
    ax.set_title(f"Visualisasi 5 Pengguna dari Sampel Terbaik (ID: {sample_id})", fontsize=16)
    ax.set_xlabel("Posisi X", fontsize=12)
    ax.set_ylabel("Posisi Y", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)

    # Menampilkan plot
    st.pyplot(fig)
    print("✅ Visualisasi selesai ditampilkan.")


# ================================
# Fungsi utama untuk menjalankan semua plot
# ================================
def run_all_plots(model_results_dir, baseline_results_dir, raw_data_dir, r_min):
    # Mengambil jumlah pengguna dan jumlah satelit dari config.py
    user_count = UT_COUNT
    
    # Menjalankan plot lainnya
    plot_cdf(model_results_dir, baseline_results_dir)
    plot_avg_total_power(model_results_dir, baseline_results_dir)
    plot_avg_power_per_sat(model_results_dir, baseline_results_dir)
    plot_qos_comparison(model_results_dir, baseline_results_dir)
    plot_total_network_throughput(model_results_dir, baseline_results_dir)
    
    best_sample_row = None
    best_sample_qos_count = -1
    best_sample_rate_total = -1.0
    best_sample_satellite_config = 0

    print("--- Memulai Proses Analisis: Mencari sample_id terbaik dari semua file ---")

    # Loop untuk setiap file konfigurasi satelit
    for satellite_count in SATELLITE_LIST:
        print(f"📊 Menganalisis file untuk {satellite_count} satelit...")
        try:
            file_path = os.path.join(model_results_dir, f"test_result_{satellite_count}sat.csv")
            if not os.path.exists(file_path):
                print(f"   -> ⚠️ File tidak ditemukan: {file_path}. Dilewati.")
                continue
                
            sample_data = pd.read_csv(file_path)

            # Loop untuk setiap baris (sample_id) di dalam file saat ini
            for index, row in sample_data.iterrows():
                current_qos_count = row['qos_count']
                current_rate_total = row['rate_total']

                # Logika perbandingan untuk menemukan baris terbaik
                # Prioritas 1: qos_count tertinggi
                if current_qos_count > best_sample_qos_count:
                    best_sample_qos_count = current_qos_count
                    best_sample_rate_total = current_rate_total
                    best_sample_row = row
                    best_sample_satellite_config = satellite_count
                # Prioritas 2: Jika qos_count sama, cek rate_total
                elif current_qos_count == best_sample_qos_count and current_rate_total > best_sample_rate_total:
                    best_sample_rate_total = current_rate_total
                    best_sample_row = row
                    best_sample_satellite_config = satellite_count

        except Exception as e:
            print(f"   -> ❌ Terjadi error saat memproses file untuk {satellite_count} satelit: {e}")

    # --- Pemicu Visualisasi Setelah Semua File Dianalisis ---
    print("\n--- Analisis Selesai ---")

    if best_sample_row is not None:
        print(f"🏆 Ditemukan sample_id terbaik di index: {best_sample_row.name}")
        print(f"   -> Dari file konfigurasi: {best_sample_satellite_config} satelit")
        print(f"   -> Dengan qos_count: {best_sample_qos_count} dan rate_total: {best_sample_rate_total:.2f}")

        # Panggil fungsi plot dengan data baris terbaik yang sudah ditemukan
        plot_5_users_from_best_sample(
            satellite_count=best_sample_satellite_config,
            best_sample_row=best_sample_row,
            R_MIN=R_MIN
        )
    else:
        print("❌ Tidak ada data sampel yang valid yang berhasil dianalisis untuk ditampilkan.")

