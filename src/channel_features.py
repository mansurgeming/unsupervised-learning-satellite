# src/channel_features_wide.py

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skyfield.api import load, wgs84

# --- KONFIGURASI ---
N_SATS_LIST = [1, 2, 4, 6, 8] # Skenario yang akan diproses
BASE_RAW_PATH = "data/raw/"
SAVE_DIR = "data/processed"
N_UT = 10
VAL_RATIO = 0.2

def compute_single_link_features(ut_pos, sat_pos, ts, t, fc=20e9, c=3e8, eta=2.0, sigma_shadow=2.0):
    """Menghitung fitur untuk satu link UT-Satelit."""
    ut_lat, ut_lon = ut_pos
    sat_lat, sat_lon, sat_alt = sat_pos
    
    sat = wgs84.latlon(sat_lat, sat_lon, elevation_m=sat_alt * 1000)
    ut = wgs84.latlon(ut_lat, ut_lon)
    
    elev, azim, dist = (sat - ut).at(t).altaz()

    dist_km = dist.km
    elev_deg = elev.degrees
    
    # Hitung komponen redaman
    dist_m = dist.m
    fspl = 20 * np.log10(4 * np.pi * dist_m * fc / c)
    shadow = np.random.normal(0, sigma_shadow)
    angle_const = (32 * np.log(2)) / (2 * (2 * np.arccos(np.sqrt(0.5)))**2)
    angle_term = (np.cos(np.radians(elev_deg)) ** eta) * angle_const
    angle_loss = -10 * np.log10(angle_term + 1e-12)
    path_loss_total = fspl + shadow + angle_loss

    # Kembalikan semua fitur yang relevan
    return {
        "dist_km": dist_km,
        "elev_deg": elev_deg,
        "azim_deg": azim.degrees,
        "fspl_db": fspl,
        "shadowing_db": shadow,
        "angle_loss_db": angle_loss,
        "path_loss_db": path_loss_total
    }

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = load.timescale()
    t = ts.now()

    for n_sats in N_SATS_LIST:
        # Gunakan file dari data lingkaran yang baru dibuat
        raw_path = os.path.join(BASE_RAW_PATH, f"raw_data_{n_sats}satelite_circle.csv")
        
        if not os.path.exists(raw_path):
            print(f"[PERINGATAN] File tidak ditemukan: {raw_path}. Melewati skenario {n_sats} satelit.")
            continue

        print(f"\n[INFO] Memulai proses untuk {n_sats} satelit dari file: {raw_path}")
        df = pd.read_csv(raw_path)
        
        all_ut_rows = []
        grouped = df.groupby("sample_id")
        
        for sample_id, group in tqdm(grouped, desc=f"Memproses Sampel ({n_sats} Satelit)"):
            ut_positions = group[["ut_lat", "ut_lon"]].values

            # Ekstrak posisi untuk semua satelit dalam sampel ini
            sat_positions = []
            for i in range(1, n_sats + 1):
                sat_cols = [f"sat_{i}_lat", f"sat_{i}_lon", f"sat_{i}_alt_km"]
                sat_pos = group.iloc[0][sat_cols].values
                sat_positions.append(sat_pos)
            
            # Buat satu baris lebar untuk setiap UT
            for ut_id in range(N_UT):
                ut_pos = ut_positions[ut_id]
                
                # Buat baris data dasar
                wide_row = {
                    "sample_id": sample_id,
                    "ut_id": ut_id,
                }
                
                # Hitung fitur untuk setiap satelit dan tambahkan ke baris
                for sat_id_idx, sat_pos in enumerate(sat_positions):
                    sat_id = sat_id_idx + 1 # ID Satelit mulai dari 1
                    
                    features = compute_single_link_features(ut_pos, sat_pos, ts, t)
                    
                    # Tambahkan fitur ke baris dengan nama kolom yang unik
                    for key, value in features.items():
                        wide_row[f"{key}_sat_{sat_id}"] = value
                
                all_ut_rows.append(wide_row)

        full_df = pd.DataFrame(all_ut_rows)
        print(f"[INFO] Total baris yang dihasilkan untuk {n_sats} satelit: {len(full_df)}")

        sample_ids = full_df["sample_id"].unique()
        train_ids, val_ids = train_test_split(sample_ids, test_size=VAL_RATIO, random_state=42)

        train_df = full_df[full_df["sample_id"].isin(train_ids)].reset_index(drop=True)
        val_df = full_df[full_df["sample_id"].isin(val_ids)].reset_index(drop=True)
        
        train_df.to_csv(os.path.join(SAVE_DIR, f"train_data_wide_{n_sats}sat.csv"), index=False)
        val_df.to_csv(os.path.join(SAVE_DIR, f"val_data_wide_{n_sats}sat.csv"), index=False)

        print(f"[SELESAI] Menyimpan {len(train_df)} baris train dan {len(val_df)} baris val untuk skenario {n_sats} satelit.")

if __name__ == "__main__":
    main()