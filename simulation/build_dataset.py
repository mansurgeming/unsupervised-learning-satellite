import os
import csv
import numpy as np
from tqdm import tqdm
from skyfield.api import load
import sys

# ================================================
# KONFIGURASI DEFAULT
# ================================================
# Nilai-nilai ini digunakan jika file dijalankan secara mandiri
DEFAULT_REGION_CENTER_LAT = 33.0
DEFAULT_REGION_CENTER_LON = 120.0
DEFAULT_AREA_RADIUS_KM = 500
DEFAULT_SAT_ALTITUDE_KM = 550
TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"

# ================================================
# FUNGSI-FUNGSI PEMBANTU (HELPER FUNCTIONS)
# ================================================

def generate_random_circular_positions(center_lat, center_lon, radius_km, num_points):
    """Menghasilkan koordinat acak dalam bentuk lingkaran."""
    lat_degree_per_km = 1.0 / 111.0
    lon_degree_per_km = 1.0 / (111.0 * np.cos(np.radians(center_lat)))
    random_radii = np.sqrt(np.random.uniform(0, 1, num_points)) * radius_km
    random_angles = np.random.uniform(0, 2 * np.pi, num_points)
    delta_lat_km = random_radii * np.cos(random_angles)
    delta_lon_km = random_radii * np.sin(random_angles)
    delta_lat_deg = delta_lat_km * lat_degree_per_km
    delta_lon_deg = delta_lon_km * lon_degree_per_km
    lats = center_lat + delta_lat_deg
    lons = center_lon + delta_lon_deg
    return np.column_stack((lats, lons))

def generate_random_square_positions(center_lat, center_lon, side_km, num_points):
    """Menghasilkan koordinat acak dalam bentuk kotak."""
    lat_degree_per_km = 1.0 / 111.0
    lon_degree_per_km = 1.0 / (111.0 * np.cos(np.radians(center_lat)))
    half_side_lat = (side_km / 2.0) * lat_degree_per_km
    half_side_lon = (side_km / 2.0) * lon_degree_per_km
    min_lat, max_lat = center_lat - half_side_lat, center_lat + half_side_lat
    min_lon, max_lon = center_lon - half_side_lon, center_lon + half_side_lon
    lats = np.random.uniform(min_lat, max_lat, num_points)
    lons = np.random.uniform(min_lon, max_lon, num_points)
    return np.column_stack((lats, lons))

def get_realistic_altitude():
    """Mendapatkan ketinggian satelit dari TLE atau menggunakan default."""
    print("[INFO] Mengunduh data TLE untuk mendapatkan ketinggian satelit...")
    try:
        ts = load.timescale()
        t = ts.now()
        sats = load.tle_file(TLE_URL, reload=True) # Paksa reload
        reference_sat = sats[0]
        altitude = reference_sat.at(t).subpoint().elevation.km
        print(f"[INFO] Ketinggian referensi dari satelit '{reference_sat.name}' adalah {altitude:.2f} km.")
        return altitude
    except Exception as e:
        print(f"[PERINGATAN] Gagal mengunduh data TLE: {e}. Menggunakan ketinggian default.")
        return DEFAULT_SAT_ALTITUDE_KM

# ================================================
# FUNGSI UTAMA (MAIN FUNCTION)
# ================================================

def run(n_sats_list, n_samples, save_dir, ut_count=5):
    """
    Fungsi utama untuk membuat dataset mentah.
    Bisa dipanggil dari skrip lain (misalnya, app.py).
    """
    print(f"--- Memulai Proses Pembuatan Dataset ---")
    print(f"Direktori penyimpanan: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    sat_altitude_km = get_realistic_altitude()

    for n_sats in n_sats_list:
        save_path = os.path.join(save_dir, f"test_data_{n_sats}sats.csv")
        print(f"\n[INFO] Membuat data untuk {n_sats} satelit ({n_samples} sampel)...")

        header = ["sample_id", "ut_id", "ut_lat", "ut_lon"]
        for i in range(1, n_sats + 1):
            header.extend([f"sat_{i}_lat", f"sat_{i}_lon", f"sat_{i}_alt_km"])

        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            # Menggunakan tqdm untuk progress bar di terminal
            for sample_id in tqdm(range(n_samples), desc=f"Data {n_sats} Sat", file=sys.stdout):
                ut_positions = generate_random_circular_positions(
                    DEFAULT_REGION_CENTER_LAT, DEFAULT_REGION_CENTER_LON, DEFAULT_AREA_RADIUS_KM, ut_count
                )
                sat_positions = generate_random_square_positions(
                    DEFAULT_REGION_CENTER_LAT, DEFAULT_REGION_CENTER_LON, 2000, n_sats
                )

                sat_data_flat = [val for pos in sat_positions for val in (*pos, sat_altitude_km)]

                for ut_id, (ut_lat, ut_lon) in enumerate(ut_positions):
                    row = [sample_id, ut_id, ut_lat, ut_lon] + sat_data_flat
                    writer.writerow(row)

        print(f"✅ [SELESAI] Data untuk {n_sats} satelit disimpan di: {save_path}")
    print("\n--- Semua Proses Pembuatan Dataset Selesai ---")

# ================================================
# BLOK EKSEKUSI (JIKA FILE DIJALANKAN LANGSUNG)
# ================================================

if __name__ == "__main__":
    # Blok ini hanya akan berjalan jika Anda menjalankan `python build_dataset.py`
    # Ini berguna untuk pengujian mandiri.
    
    print("Menjalankan build_dataset.py sebagai skrip mandiri...")
    
    # Contoh konfigurasi untuk pengujian
    N_SATS_TO_TEST = [2, 6]
    N_SAMPLES_TO_TEST = 50 # Jumlah sampel lebih kecil untuk tes cepat
    SAVE_DIRECTORY_TEST = "data/raw" # Simpan ke subfolder lokal
    UT_COUNT_TEST = 5

    # Memanggil fungsi utama dengan konfigurasi tes
    run(
        n_sats_list=N_SATS_TO_TEST,
        n_samples=N_SAMPLES_TO_TEST,
        save_dir=SAVE_DIRECTORY_TEST,
        ut_count=UT_COUNT_TEST
    )