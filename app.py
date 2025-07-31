# Nama file: app.py

import streamlit as st
import os
import sys

# --- 1. SETUP PATH DAN IMPOR ---
script_dir = os.path.dirname(os.path.abspath(__file__))
simulation_path = os.path.join(script_dir, 'simulation')
if simulation_path not in sys.path:
    sys.path.append(simulation_path)

try:
    # Impor SATU FUNGSI UTAMA dari file plotting Anda
    from plotting_dashboard import run_all_plots 
    from config import R_MIN
    print("✅ Modul plotting dan konfigurasi berhasil diimpor.")
except ImportError as e:
    st.error(f"❌ Gagal mengimpor dari 'plotting_dashboard.py' atau 'config.py'.")
    st.code(f"Detail Error: {e}")
    st.stop()

# --- 2. KONFIGURASI HALAMAN DAN KONTEN UTAMA ---
st.set_page_config(
    page_title="Dashboard Simulasi Satelit",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Dashboard Hasil Simulasi Alokasi Daya Satelit")

# Definisikan path ke folder-folder penting
model_results_dir = os.path.join(script_dir, "results", "model_results")
baseline_results_dir = os.path.join(script_dir, "results", "baseline_results")
raw_data_dir = os.path.join(script_dir, "data", "raw")

# --- 3. JALANKAN SEMUA PLOT ---
# Memanggil satu fungsi utama yang akan mengatur semua tampilan plot
run_all_plots(
    model_results_dir=model_results_dir,
    baseline_results_dir=baseline_results_dir,
    raw_data_dir=raw_data_dir,
    r_min=R_MIN
)

# --- Footer ---
st.divider()
st.markdown("Dibuat untuk Proyek Simulasi Satelit.")