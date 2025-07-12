# src/build_raw_dataset.py

import numpy as np
import os
import csv
from tqdm import tqdm
from skyfield.api import load, wgs84
from geopy.distance import distance
from geopy.point import Point

REGION_CENTER = (31.5, 121.0)
SAVE_PATH = "data/raw/raw_dataset.csv"
N_SAMPLES = 10000
N_UT = 10
UT_RADIUS_KM = 500

def generate_random_utm_positions(center_latlon, size_km, num_points):
    center = Point(center_latlon)
    radius_km = size_km / 2
    positions = []
    for _ in range(num_points):
        bearing = np.random.uniform(0, 360)
        dist = np.random.uniform(0, radius_km)
        pt = distance(kilometers=dist).destination(center, bearing)
        positions.append((pt.latitude, pt.longitude))
    return np.array(positions)

def main():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    print("[INFO] Loading TLE...")
    ts = load.timescale()
    t = ts.now()
    sats = load.tle_file("https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle")

    sat = sats[0]
    subpoint = sat.at(t).subpoint()
    sat_lat, sat_lon, sat_alt_km = (
        subpoint.latitude.degrees,
        subpoint.longitude.degrees,
        subpoint.elevation.km,
    )
    print(f"[INFO] Satelit: {sat.name} @ {sat_lat:.2f}, {sat_lon:.2f}, alt {sat_alt_km:.2f} km")

    with open(SAVE_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "ut_id", "ut_lat", "ut_lon", "sat_lat", "sat_lon", "sat_alt_km"])

        for i in tqdm(range(N_SAMPLES)):
            ut_positions = generate_random_utm_positions(REGION_CENTER, UT_RADIUS_KM, N_UT)
            for ut_id, (lat, lon) in enumerate(ut_positions):
                writer.writerow([i, ut_id, lat, lon, sat_lat, sat_lon, sat_alt_km])

    print(f"[DONE] Dataset saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()