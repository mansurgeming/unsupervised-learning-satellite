import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def plot_best_qos_sample_layout(m_choice, model_results_dir, raw_data_dir):
    """
    Mencari sampel dengan QoS terbaik untuk sejumlah satelit yang dipilih,
    dan memvisualisasikan tata letak geografisnya.

    Args:
        m_choice (int): Jumlah satelit yang dipilih untuk ditampilkan.
        model_results_dir (str): Path ke direktori hasil tes model.
        raw_data_dir (str): Path ke direktori data mentah (yang berisi posisi).

    Returns:
        matplotlib.figure.Figure: Objek figure dari plot yang dihasilkan,
                                  atau None jika terjadi error.
    """
    try:
        # 1. Cari sampel dengan QoS terbaik dari file hasil
        results_path = os.path.join(model_results_dir, f"test_result_{m_choice}sat.csv")
        df_results = pd.read_csv(results_path)
        
        if df_results.empty:
            print(f"File hasil untuk {m_choice} satelit kosong.")
            return None

        # Temukan baris dengan 'qos_count' tertinggi
        best_row = df_results.loc[df_results['qos_count'].idxmax()]
        best_sample_id = int(best_row['sample_id'])
        best_qos_count = int(best_row['qos_count'])

        # 2. Ambil data posisi dari file mentah menggunakan sample_id terbaik
        raw_path = os.path.join(raw_data_dir, f"test_data_{m_choice}sats.csv")
        df_raw = pd.read_csv(raw_path)
        sample_df = df_raw[df_raw['sample_id'] == best_sample_id]

        if sample_df.empty:
            print(f"Data posisi untuk sample_id {best_sample_id} tidak ditemukan.")
            return None

        # 3. Ekstrak koordinat UT dan Satelit
        # Ambil posisi unik karena setiap baris di sampel mentah memiliki posisi UT yang sama
        ut_positions = sample_df[['ut_lon', 'ut_lat']].drop_duplicates()
        ut_lons = ut_positions['ut_lon']
        ut_lats = ut_positions['ut_lat']
        
        # Ambil posisi satelit dari baris pertama sampel (semua sama)
        first_row = sample_df.iloc[0]
        sat_lats = [first_row[f'sat_{i}_lat'] for i in range(1, m_choice + 1)]
        sat_lons = [first_row[f'sat_{i}_lon'] for i in range(1, m_choice + 1)]

        # 4. Buat plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(ut_lons, ut_lats, marker='o', color='blue', s=50, label='Users (UT)')
        ax.scatter(sat_lons, sat_lats, marker='^', color='red', s=150, edgecolors='black', label='Satellites')
        
        ax.set_title(f"Tata Letak Sampel QoS Terbaik ({m_choice} Satelit)\nID: {best_sample_id} | QoS Tercapai: {best_qos_count}/{len(ut_positions)}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()
        
        return fig

    except FileNotFoundError:
        print(f"❌ Error: File yang dibutuhkan untuk plot tata letak tidak ditemukan.")
        print(f"Pastikan 'test_result_{m_choice}sat.csv' dan 'test_data_{m_choice}sats.csv' ada.")
        return None
    except Exception as e:
        print(f"❌ Terjadi kesalahan saat membuat plot tata letak: {e}")
        return None

# --- Blok Eksekusi Mandiri ---
if __name__ == '__main__':
    print("Menjalankan position_plot.py sebagai skrip mandiri...")

    try:
        # Menambahkan path root proyek agar bisa mengimpor config.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        sys.path.append(project_root)
        
        # Impor konfigurasi secara dinamis
        from config import SATELLITE_LIST, RESULTS_MODEL_DIR, RAW_DATA_DIR
        
        M_LIST_TEST = SATELLITE_LIST
        MODEL_DIR_TEST = RESULTS_MODEL_DIR
        RAW_DIR_TEST = RAW_DATA_DIR

    except ImportError:
        print("❌ Gagal mengimpor dari config.py. Menggunakan nilai default.")
        M_LIST_TEST = [1, 6]
        MODEL_DIR_TEST = "results/model_results"
        RAW_DIR_TEST = "data/raw"

    # --- Hasilkan plot untuk setiap konfigurasi satelit ---
    for m_choice in M_LIST_TEST:
        print(f"\n--- Membuat plot untuk {m_choice} satelit ---")
        fig = plot_best_qos_sample_layout(
            m_choice=m_choice,
            model_results_dir=MODEL_DIR_TEST,
            raw_data_dir=RAW_DIR_TEST
        )
        
        # Tampilkan plot jika berhasil dibuat
        if fig:
            plt.show()