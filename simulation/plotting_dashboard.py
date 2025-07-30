# Nama file: simulation/plotting_dashboard.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import streamlit as st
import re
import sys

# ================================================
# Fungsi 1: Plot CDF (dari cdf_plot.py)
# ================================================
def plot_cdf(base_dir, model_dir, m_list_options, r_min):
    st.subheader("1. CDF Data Rate per Pengguna (UT)")
    m_choice = st.selectbox(
        "Pilih konfigurasi satelit untuk melihat CDF:",
        options=m_list_options,
        key="cdf_selector"
    )
    if not m_choice: return

    fig, ax = plt.subplots(figsize=(8, 5))
    baseline_path = os.path.join(base_dir, f"test_result_{m_choice}sat.csv")
    model_path = os.path.join(model_dir, f"test_result_{m_choice}sat.csv")

    if not os.path.exists(baseline_path) or not os.path.exists(model_path):
        st.warning(f"Data CDF untuk {m_choice} satelit tidak ditemukan."); return

    def load_rates(path):
        df = pd.read_csv(path)
        all_rates = [rate for r_str in df['rate_per_ut'] for rate in json.loads(r_str)]
        all_rates_sorted = np.sort(all_rates)
        return all_rates_sorted, np.arange(1, len(all_rates_sorted) + 1) / len(all_rates_sorted)

    x_base, y_base = load_rates(baseline_path)
    x_model, y_model = load_rates(model_path)

    ax.plot(x_base, y_base, label="Baseline", linestyle='--'); ax.plot(x_model, y_model, label="Model")
    ax.axvline(x=r_min, color='red', linestyle=':', linewidth=1.5, label=f"R_min={r_min}")
    ax.set_title(f"CDF of Data Rate ({m_choice} Satelit)"); ax.set_xlabel("Rate per UT (bps/Hz)")
    ax.set_ylabel("CDF"); ax.grid(True, alpha=0.5); ax.legend(); st.pyplot(fig)

# ================================================
# Fungsi 2: Plot Perbandingan QoS (dari qos_comparison_plot.py)
# ================================================
def plot_qos_comparison(base_dir, model_dir, m_list, num_ut):
    st.subheader("2. Rata-rata Tingkat Kepuasan QoS")
    data = []
    for M in m_list:
        try:
            df_model = pd.read_csv(os.path.join(model_dir, f"test_result_{M}sat.csv"))
            df_baseline = pd.read_csv(os.path.join(base_dir, f"test_result_{M}sat.csv"))
            avg_model = (df_model["qos_count"] / num_ut).mean()
            avg_baseline = (df_baseline["qos_count"] / num_ut).mean()
            data.append({"Satelit": str(M), "Model": avg_model, "Baseline": avg_baseline})
        except (FileNotFoundError, pd.errors.EmptyDataError):
            st.warning(f"Data QoS untuk {M} satelit tidak ditemukan atau kosong.")
    if data:
        df_qos = pd.DataFrame(data).set_index("Satelit"); st.bar_chart(df_qos)
        st.caption("Grafik menunjukkan fraksi rata-rata pengguna yang memenuhi target QoS (nilai 1.0 = 100%).")

# ================================================
# Fungsi 3: Plot Rata-rata TOTAL Daya Transmit (dari avg_transmit_plot.py)
# ================================================
def plot_avg_total_power(base_dir, model_dir, m_list):
    st.subheader("3. Rata-rata Total Daya Transmit (Seluruh Sistem)")
    data = []
    for M in m_list:
        try:
            df_model = pd.read_csv(os.path.join(model_dir, f"test_result_{M}sat.csv"))
            avg_power_model = df_model["power_per_sat"].apply(lambda x: sum(json.loads(x))).mean()
            df_baseline = pd.read_csv(os.path.join(base_dir, f"test_result_{M}sat.csv"))
            avg_power_baseline = df_baseline["power_per_sat"].apply(lambda x: sum(json.loads(x))).mean()
            data.append({"Satelit": str(M), "Model": avg_power_model, "Baseline": avg_power_baseline})
        except (FileNotFoundError, pd.errors.EmptyDataError):
            st.warning(f"Data Total Daya untuk {M} satelit tidak ditemukan atau kosong.")
    if data:
        df_power = pd.DataFrame(data).set_index("Satelit"); st.bar_chart(df_power)
        st.caption("Grafik menunjukkan total daya rata-rata yang dikeluarkan oleh semua satelit dalam satu waktu (Watt).")

# ================================================
# Fungsi 4: Plot Rata-rata Daya PER SATELIT (dari avg_transmit_per_satelite_plot.py)
# ================================================
def plot_avg_power_per_satellite(base_dir, model_dir, m_list):
    st.subheader("4. Rata-rata Daya yang Digunakan per Satelit")
    data = []
    for M in m_list:
        try:
            df_model = pd.read_csv(os.path.join(model_dir, f"test_result_{M}sat.csv"))
            avg_total_power_model = df_model["power_per_sat"].apply(lambda x: sum(json.loads(x))).mean()
            avg_per_satellite_model = avg_total_power_model / M
            df_baseline = pd.read_csv(os.path.join(base_dir, f"test_result_{M}sat.csv"))
            avg_total_power_baseline = df_baseline["power_per_sat"].apply(lambda x: sum(json.loads(x))).mean()
            avg_per_satellite_baseline = avg_total_power_baseline / M
            data.append({"Satelit": str(M), "Model": avg_per_satellite_model, "Baseline": avg_per_satellite_baseline})
        except (FileNotFoundError, pd.errors.EmptyDataError, ZeroDivisionError):
            st.warning(f"Data Daya per Satelit untuk {M} satelit tidak ditemukan/valid.")
    if data:
        df_power = pd.DataFrame(data).set_index("Satelit"); st.bar_chart(df_power)
        st.caption("Grafik menunjukkan daya rata-rata yang dikeluarkan oleh satu satelit (Watt).")

# ================================================
# Fungsi 5: Plot Total Network Throughput (dari total_network_throughput_plot.py)
# ================================================
def plot_avg_network_throughput(base_dir, model_dir, m_list):
    st.subheader("5. Rata-rata Total Network Throughput")
    def compute_avg_throughput(file_path):
        df = pd.read_csv(file_path)
        total_per_sample = [sum(sorted(json.loads(r_str), reverse=True)[:int(qos_count)]) for r_str, qos_count in zip(df["rate_per_ut"], df["qos_count"])]
        return np.mean(total_per_sample)
    data = []
    for M in m_list:
        try:
            avg_throughput_model = compute_avg_throughput(os.path.join(model_dir, f"test_result_{M}sat.csv"))
            avg_throughput_baseline = compute_avg_throughput(os.path.join(base_dir, f"test_result_{M}sat.csv"))
            data.append({"Satelit": str(M), "Model": avg_throughput_model, "Baseline": avg_throughput_baseline})
        except (FileNotFoundError, pd.errors.EmptyDataError):
            st.warning(f"Data Throughput untuk {M} satelit tidak ditemukan atau kosong.")
    if data:
        df_throughput = pd.DataFrame(data).set_index("Satelit"); st.bar_chart(df_throughput)
        st.caption("Grafik menunjukkan jumlah total data rate rata-rata dari pengguna yang memenuhi target QoS (bps/Hz).")

# ================================================
# Fungsi 6: Plot Tata Letak Sampel QoS Terbaik (dari position_plot.py)
# ================================================
def plot_best_qos_sample_layout(model_results_dir, raw_data_dir, m_choice):
    try:
        results_path = os.path.join(model_results_dir, f"test_result_{m_choice}sat.csv")
        df_results = pd.read_csv(results_path)
        if df_results.empty: st.warning("File hasil kosong."); return
        best_row = df_results.loc[df_results['qos_count'].idxmax()]
        best_sample_id = int(best_row['sample_id'])
        best_qos_count = int(best_row['qos_count'])

        raw_path = os.path.join(raw_data_dir, f"test_data_{m_choice}sats.csv")
        df_raw = pd.read_csv(raw_path)
        sample_df = df_raw[df_raw['sample_id'] == best_sample_id]
        if sample_df.empty: st.warning(f"Data posisi untuk sample_id {best_sample_id} tidak ditemukan."); return
        
        ut_positions = sample_df[['ut_lon', 'ut_lat']].drop_duplicates()
        first_row = sample_df.iloc[0]
        sat_lons = [first_row[f'sat_{i}_lon'] for i in range(1, m_choice + 1)]
        sat_lats = [first_row[f'sat_{i}_lat'] for i in range(1, m_choice + 1)]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(ut_positions['ut_lon'], ut_positions['ut_lat'], marker='o', color='blue', s=50, label='Users (UT)')
        ax.scatter(sat_lons, sat_lats, marker='^', color='red', s=150, edgecolors='black', label='Satellites')
        ax.set_title(f"Tata Letak Sampel QoS Terbaik ({m_choice} Satelit)\nID: {best_sample_id} | QoS: {best_qos_count}/{len(ut_positions)}")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal', adjustable='box'); fig.tight_layout(); st.pyplot(fig)
    except FileNotFoundError: st.error(f"File yang dibutuhkan untuk plot tata letak tidak ditemukan.")
    except Exception as e: st.error(f"Terjadi kesalahan saat membuat plot tata letak: {e}")

# ================================================
# Fungsi Utama untuk Menjalankan Semua Plot
# ================================================
def run_all_plots(model_results_dir, baseline_results_dir, raw_data_dir, r_min):
    st.header("Visualisasi Hasil Analisis Batch")

    if not os.path.exists(model_results_dir) or not os.path.exists(baseline_results_dir):
        st.error(f"Direktori hasil tidak ditemukan. Jalankan `master_runner.py` terlebih dahulu."); return

    available_sats = []
    for f in os.listdir(model_results_dir):
        if f.startswith("test_result_") and f.endswith("sat.csv"):
            try:
                num = int(re.search(r'_(\d+)sat\.csv', f).group(1))
                if os.path.exists(os.path.join(baseline_results_dir, f)): available_sats.append(num)
            except (AttributeError, ValueError): continue
    available_sats = sorted(list(set(available_sats)))
    if not available_sats: st.warning("Tidak ada file hasil pengujian yang cocok ditemukan."); return

    st.markdown("### Plot Agregat (Perbandingan Kinerja Rata-rata)")
    m_list_agregat = st.multiselect("Pilih konfigurasi satelit untuk plot perbandingan:", options=available_sats, default=available_sats)
    
    if m_list_agregat:
        NUM_UT = 5
        plot_qos_comparison(baseline_results_dir, model_results_dir, m_list_agregat, NUM_UT)
        st.divider()
        plot_avg_total_power(baseline_results_dir, model_results_dir, m_list_agregat)
        st.divider()
        plot_avg_power_per_satellite(baseline_results_dir, model_results_dir, m_list_agregat)
        st.divider()
        plot_avg_network_throughput(baseline_results_dir, model_results_dir, m_list_agregat)

    st.divider()
    st.markdown("### Analisis Detail per Konfigurasi")
    plot_cdf(baseline_results_dir, model_results_dir, available_sats, r_min)
    st.divider()
    with st.expander("Lihat Tata Letak Geografis Sampel Terbaik"):
        m_choice_layout = st.selectbox("Pilih konfigurasi untuk dilihat:", options=available_sats)
        if m_choice_layout: plot_best_qos_sample_layout(model_results_dir, raw_data_dir, m_choice_layout)

# ================================================
# Blok Eksekusi Mandiri (untuk testing)
# ================================================
if __name__ == '__main__':
    st.info("Menjalankan `plotting_dashboard.py` dalam mode mandiri untuk debug.")
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        sys.path.append(project_root)
        from config import R_MIN_THRESHOLD, RESULTS_MODEL_DIR, RESULTS_BASELINE_DIR, RAW_DATA_DIR
        run_all_plots(RESULTS_MODEL_DIR, RESULTS_BASELINE_DIR, RAW_DATA_DIR, R_MIN_THRESHOLD)
    except ImportError:
        st.error("Gagal memuat `config.py`. Pastikan file config ada di folder root.")