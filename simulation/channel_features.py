import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from skyfield.api import load, wgs84

# ================================================
# FUNGSI PEMBANTU (HELPER FUNCTION)
# ================================================

def compute_ut_sat_features(ut_lat, ut_lon, sat_lat, sat_lon, sat_alt_km, t,
                             fc=12e9, c=3e8, eta=2.0, sigma_shadow=2.0):
    """Menghitung semua fitur channel antara satu UT dan satu satelit."""
    ut = wgs84.latlon(ut_lat, ut_lon)
    sat = wgs84.latlon(sat_lat, sat_lon, elevation_m=sat_alt_km * 1000)
    elev, azim, dist = (sat - ut).at(t).altaz()

    dist_m = dist.m
    elev_deg = elev.degrees

    # Kalkulasi Path Loss Components
    fspl = 20 * np.log10(4 * np.pi * dist_m * fc / c)
    shadow = np.random.normal(0, sigma_shadow)
    angle_const = (32 * np.log(2)) / (2 * (2 * np.arccos(np.sqrt(0.5)))**2)
    angle_term = (np.cos(np.radians(elev_deg)) ** eta) * angle_const
    angle_loss = -10 * np.log10(angle_term + 1e-12)
    path_loss_total = fspl + shadow + angle_loss

    return {
        "dist_km": dist.km,
        "elev_deg": elev_deg,
        "azim_deg": azim.degrees,
        "fspl_db": fspl,
        "shadowing_db": shadow,
        "angle_loss_db": angle_loss,
        "path_loss_db": path_loss_total
    }

# ================================================
# FUNGSI UTAMA (MAIN FUNCTION)
# ================================================

def run(n_sats_list, raw_dir, processed_dir, ut_count=5):
    """
    Fungsi utama untuk memproses data mentah menjadi data fitur channel.
    Membaca file dari `raw_dir` dan menyimpan hasilnya ke `processed_dir`.
    """
    print(f"--- Memulai Proses Perhitungan Fitur Channel ---")
    os.makedirs(processed_dir, exist_ok=True)
    
    ts = load.timescale()
    t = ts.now()

    for n_sats in n_sats_list:
        raw_path = os.path.join(raw_dir, f"test_data_{n_sats}sats.csv")
        
        if not os.path.exists(raw_path):
            print(f"⚠️  [PERINGATAN] File data mentah tidak ditemukan: {raw_path}. Melewati...")
            continue
            
        print(f"\n[INFO] Memuat data mentah dari: {raw_path}")
        df = pd.read_csv(raw_path)

        rows = []
        # Menggunakan tqdm untuk progress bar di terminal
        for sample_id, group in tqdm(df.groupby("sample_id"), desc=f"Fitur {n_sats} Sat", file=sys.stdout):
            ut_positions = group[["ut_lat", "ut_lon"]].values
            
            # Ekstrak posisi satelit sekali per sampel untuk efisiensi
            sat_positions = []
            for m in range(1, n_sats + 1):
                sat_lat = group.iloc[0][f"sat_{m}_lat"]
                sat_lon = group.iloc[0][f"sat_{m}_lon"]
                sat_alt = group.iloc[0][f"sat_{m}_alt_km"]
                sat_positions.append((m, sat_lat, sat_lon, sat_alt))

            for u_id in range(ut_count):
                ut_lat, ut_lon = ut_positions[u_id]
                for m, sat_lat, sat_lon, sat_alt in sat_positions:
                    # Hitung fitur untuk setiap pasangan UT-Satelit
                    feat = compute_ut_sat_features(ut_lat, ut_lon, sat_lat, sat_lon, sat_alt, t)
                    rows.append({
                        "sample_id": sample_id,
                        "ut_id": u_id,
                        "sat_id": m,
                        **feat
                    })

        # Simpan hasil ke file CSV di direktori yang diproses
        test_df = pd.DataFrame(rows)
        out_path = os.path.join(processed_dir, f"test_data_{n_sats}sat.csv")
        test_df.to_csv(out_path, index=False)
        print(f"✅ [SELESAI] {len(test_df)} baris data fitur disimpan di: {out_path}")
    print("\n--- Semua Proses Perhitungan Fitur Selesai ---")

# ================================================
# BLOK EKSEKUSI (JIKA FILE DIJALANKAN LANGSUNG)
# ================================================

if __name__ == "__main__":
    # Blok ini hanya akan berjalan jika Anda menjalankan `python channel_features.py`
    print("Menjalankan channel_features.py sebagai skrip mandiri...")

    # Contoh konfigurasi untuk pengujian
    N_SATS_TO_TEST = [1, 2] # Daftar satelit yang akan diproses
    RAW_DIR_TEST = "data/raw"
    PROCESSED_DIR_TEST = "data/processed"
    UT_COUNT_TEST = 5

    # Memanggil fungsi utama dengan konfigurasi tes
    run(
        n_sats_list=N_SATS_TO_TEST,
        raw_dir=RAW_DIR_TEST,
        processed_dir=PROCESSED_DIR_TEST,
        ut_count=UT_COUNT_TEST
    )