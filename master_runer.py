# Nama file: master_runner.py

import os
import sys
import subprocess

# --- Konfigurasi Utama Pipeline ---
SATELLITE_LIST = [1, 2, 6, 8, 10]
NUM_SAMPLES = 1000
UT_COUNT = 5
R_MIN_THRESHOLD = 0.035

def run_pipeline():
    """
    Fungsi utama untuk menjalankan seluruh pipeline simulasi.
    """
    print("🚀 ===== MEMULAI SELURUH ALUR KERJA SIMULASI ===== 🚀")

    # =================================================================
    # === BAGIAN KRUSIAL: Menambahkan folder 'simulation' ke path ===
    # =================================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    simulation_path = os.path.join(script_dir, 'simulation')
    if simulation_path not in sys.path:
        sys.path.append(simulation_path)
        print(f"[*] Path '{simulation_path}' berhasil ditambahkan ke sistem.")
    
    # Impor fungsi 'run' dari setiap modul di dalam 'simulation'
    try:
        print("[*] Mencoba mengimpor modul dari 'simulation/'...")
        from build_dataset import run as build_dataset
        from channel_features import run as calculate_features
        from run_test import run as run_model_test
        from run_baseline import run as run_baseline_test
        print("✅ Semua modul berhasil diimpor.")
    except ImportError as e:
        print("\n❌ GAGAL MENGIMPOR MODUL!")
        print("Pastikan Anda sudah membuat file '__init__.py' yang kosong di dalam folder 'simulation/'.")
        print(f"Detail Error: {e}")
        return

    # --- Definisikan direktori relatif dari folder root ---
    RAW_DATA_DIR = os.path.join(script_dir, "data", "raw")
    PROCESSED_DATA_DIR = os.path.join(script_dir, "data", "processed")
    MODELS_DIR = os.path.join(script_dir, "models")
    RESULTS_MODEL_DIR = os.path.join(script_dir, "results", "model_results")
    RESULTS_BASELINE_DIR = os.path.join(script_dir, "results", "baseline_results")

    # --- [LANGKAH 1] Membuat Dataset Mentah ---
    print("\n[LANGKAH 1/4] Membuat Dataset Mentah...")
    build_dataset(n_sats_list=SATELLITE_LIST, n_samples=NUM_SAMPLES, save_dir=RAW_DATA_DIR, ut_count=UT_COUNT)

    # --- [LANGKAH 2] Menghitung Fitur Channel ---
    print("\n[LANGKAH 2/4] Menghitung Fitur Channel...")
    calculate_features(n_sats_list=SATELLITE_LIST, raw_dir=RAW_DATA_DIR, processed_dir=PROCESSED_DATA_DIR, ut_count=UT_COUNT)

    # --- [LANGKAH 3] Menjalankan Pengujian Model & Baseline ---
    print("\n[LANGKAH 3/4] Menjalankan Pengujian Model & Baseline...")
    run_model_test(n_sats_list=SATELLITE_LIST, data_dir=PROCESSED_DATA_DIR, model_dir=MODELS_DIR, result_dir=RESULTS_MODEL_DIR, ut_count=UT_COUNT, r_min=R_MIN_THRESHOLD)
    run_baseline_test(n_sats_list=SATELLITE_LIST, data_dir=PROCESSED_DATA_DIR, result_dir=RESULTS_BASELINE_DIR, ut_count=UT_COUNT, r_min=R_MIN_THRESHOLD)

    # --- [LANGKAH 4] Menjalankan Dashboard Streamlit ---
    print("\n[LANGKAH 4/4] Membuka Dashboard Streamlit untuk menampilkan hasil...")
    app_path = os.path.join(script_dir, "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])

    print("\n✅ ===== SEMUA PROSES SELESAI ===== ✅")

if __name__ == "__main__":
    run_pipeline()