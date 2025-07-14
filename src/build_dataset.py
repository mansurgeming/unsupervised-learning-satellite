# src/build_raw_dataset_rectangle_dynamic_alt.py

import numpy as np
import os
import csv
from tqdm import tqdm
from skyfield.api import load # Mengimpor kembali library skyfield

# --- KONFIGURASI ---
# Pusat area persegi
REGION_CENTER_LAT = 31.5
REGION_CENTER_LON = 121.0

# Dimensi area dan jumlah entitas
AREA_SIDE_KM = 1000
N_SAMPLES = 10000
N_UT = 10
N_SATS_LIST = [2, 4, 6, 8]
DEFAULT_SAT_ALTITUDE_KM = 550 # Ketinggian fallback jika gagal mengambil data TLE

# URL TLE untuk grup satelit Starlink
TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"

# Lokasi penyimpanan data
SAVE_DIR = "../data/raw/"

def generate_random_square_positions(center_lat, center_lon, side_km, num_points):
    """
    Menghasilkan koordinat (lat, lon) secara acak di dalam area persegi.
    """
    lat_degree_per_km = 1.0 / 111.0
    lon_degree_per_km = 1.0 / (111.0 * np.cos(np.radians(center_lat)))
    half_side_lat = (side_km / 2.0) * lat_degree_per_km
    half_side_lon = (side_km / 2.0) * lon_degree_per_km
    min_lat = center_lat - half_side_lat
    max_lat = center_lat + half_side_lat
    min_lon = center_lon - half_side_lon
    max_lon = center_lon + half_side_lon
    lats = np.random.uniform(min_lat, max_lat, num_points)
    lons = np.random.uniform(min_lon, max_lon, num_points)
    return np.column_stack((lats, lons))

def get_realistic_altitude():
    """
    Mengambil data TLE dan mengembalikan ketinggian satelit referensi.
    """
    print("[INFO] Mengunduh data TLE untuk mendapatkan ketinggian satelit...")
    try:
        ts = load.timescale()
        t = ts.now()
        sats = load.tle_file(TLE_URL)
        
        # Ambil satu satelit sebagai referensi ketinggian
        reference_sat = sats[0]
        subpoint = reference_sat.at(t).subpoint()
        altitude = subpoint.elevation.km
        
        print(f"[INFO] Ketinggian referensi dari satelit '{reference_sat.name}' adalah {altitude:.2f} km.")
        return altitude
    except Exception as e:
        print(f"[PERINGATAN] Gagal mengunduh data TLE atau TLE kosong: {e}")
        print(f"[PERINGATAN] Menggunakan ketinggian default: {DEFAULT_SAT_ALTITUDE_KM} km.")
        return DEFAULT_SAT_ALTITUDE_KM

def main():
    """
    Fungsi utama untuk membuat dan menyimpan dataset.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Dapatkan ketinggian satelit yang realistis sekali di awal
    sat_altitude_km = get_realistic_altitude()

    # Loop untuk setiap skenario jumlah satelit
    for n_sats in N_SATS_LIST:
        save_path = os.path.join(SAVE_DIR, f"raw_data_{n_sats}satelite_rectangle.csv")
        print(f"\n[INFO] Memulai pembuatan dataset untuk {n_sats} satelit...")
        print(f"[INFO] Data akan disimpan di: {save_path}")

        # Buat header CSV
        header = ["sample_id", "ut_id", "ut_lat", "ut_lon"]
        for i in range(1, n_sats + 1):
            header.extend([f"sat_{i}_lat", f"sat_{i}_lon", f"sat_{i}_alt_km"])

        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for sample_id in tqdm(range(N_SAMPLES), desc=f"Skenario {n_sats} Satelit"):
                ut_positions = generate_random_square_positions(
                    REGION_CENTER_LAT, REGION_CENTER_LON, AREA_SIDE_KM, N_UT
                )
                sat_positions = generate_random_square_positions(
                    REGION_CENTER_LAT, REGION_CENTER_LON, AREA_SIDE_KM, n_sats
                )

                sat_data_flat = []
                for lat, lon in sat_positions:
                    # Gunakan ketinggian yang sudah didapatkan untuk semua satelit
                    sat_data_flat.extend([lat, lon, sat_altitude_km])

                for ut_id, (ut_lat, ut_lon) in enumerate(ut_positions):
                    row = [sample_id, ut_id, ut_lat, ut_lon] + sat_data_flat
                    writer.writerow(row)

        print(f"[SELESAI] Dataset untuk {n_sats} satelit berhasil dibuat.")

if __name__ == "__main__":
    main()