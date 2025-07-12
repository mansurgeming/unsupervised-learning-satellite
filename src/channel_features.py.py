import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skyfield.api import load, wgs84

RAW_PATH = "data/raw/raw_dataset.csv"
SAVE_DIR = "data/processed"
N_UT = 10
VAL_RATIO = 0.2

# === Inline Channel Feature Computation ===
def compute_channel_features(ut_positions, sat_position, ts, t,
                              fc=20e9,  # carrier freq (Hz)
                              c=3e8,    # speed of light
                              eta=2.0,  # beam pattern loss exponent
                              sigma_shadow=2.0):
    K = ut_positions.shape[0]
    M = 1
    F = 5
    features = np.zeros((M, K, F))

    sat_lat, sat_lon, sat_alt = sat_position
    sat = wgs84.latlon(sat_lat, sat_lon, elevation_m=sat_alt * 1000)
    angle_const = (32 * np.log(2)) / (2 * (2 * np.arccos(np.sqrt(0.5)))**2)

    for k in range(K):
        ut_lat, ut_lon = ut_positions[k]
        ut = wgs84.latlon(ut_lat, ut_lon)
        elev, azim, dist = (sat - ut).at(t).altaz()

        dist_km = dist.km
        dist_m = dist.m
        elev_deg = elev.degrees
        azim_deg = azim.degrees

        fspl = 20 * np.log10(4 * np.pi * dist_m * fc / c)
        shadow = np.random.normal(0, sigma_shadow)
        angle_term = (np.cos(np.radians(elev_deg)) ** eta) * angle_const
        angle_loss = -10 * np.log10(angle_term + 1e-12)

        path_loss_total = fspl + shadow + angle_loss

        features[0, k] = [dist_km, elev_deg, azim_deg, path_loss_total, shadow]

    return features

# === Main ===
def main():
    print("[INFO] Loading raw CSV...")
    df = pd.read_csv(RAW_PATH)
    os.makedirs(SAVE_DIR, exist_ok=True)

    ts = load.timescale()
    t = ts.now()

    rows = []

    grouped = df.groupby("sample_id")
    for sample_id, group in tqdm(grouped, desc="Processing samples"):
        ut_positions = group[["ut_lat", "ut_lon"]].values
        sat_pos = group.iloc[0][["sat_lat", "sat_lon", "sat_alt_km"]].values

        features = compute_channel_features(
            ut_positions, sat_pos, ts, t
        )

        for ut_id in range(N_UT):
            dist_km, elev, azim, pl, shadow = features[0, ut_id]
            dist_m = dist_km * 1000
            fspl = 20 * np.log10(4 * np.pi * dist_m * 20e9 / 3e8)
            eta = 2.0
            angle_const = (32 * np.log(2)) / (2 * (2 * np.arccos(np.sqrt(0.5)))**2)
            angle_term = (np.cos(np.radians(elev)) ** eta) * angle_const
            angle_loss = -10 * np.log10(angle_term + 1e-12)

            rows.append({
                "sample_id": sample_id,
                "ut_id": ut_id,
                "dist_km": dist_km,
                "elev_deg": elev,
                "azim_deg": azim,
                "fspl_db": fspl,
                "shadowing_db": shadow,
                "angle_loss_db": angle_loss,
                "path_loss_db": pl
            })

    full_df = pd.DataFrame(rows)
    print(f"[INFO] Total rows: {len(full_df)}")

    sample_ids = full_df["sample_id"].unique()
    train_ids, val_ids = train_test_split(sample_ids, test_size=VAL_RATIO, random_state=42)

    train_df = full_df[full_df["sample_id"].isin(train_ids)].reset_index(drop=True)
    val_df = full_df[full_df["sample_id"].isin(val_ids)].reset_index(drop=True)

    train_df.to_csv(os.path.join(SAVE_DIR, "train_data.csv"), index=False)
    val_df.to_csv(os.path.join(SAVE_DIR, "val_data.csv"), index=False)

    print(f"[DONE] Saved {len(train_df)} train rows and {len(val_df)} val rows to {SAVE_DIR}/")

if __name__ == "__main__":
    main()