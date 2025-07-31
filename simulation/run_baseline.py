import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os

local_path = os.path.dirname(os.path.abspath(__file__))  # Ini akan memberi path folder tempat file Python Anda berada


# =============================
# Konfigurasi & Hyperparam
# =============================
K = 5  # Jumlah User Terminals (UT)
M = [2, 6]  # Jumlah satelit yang akan diuji
P_MAX = 300.0  # Daya maksimum yang diizinkan
R_MIN = 0.35  # Threshold untuk QoS (Rate minimum per UT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils import compute_noise_power, compute_beamforming, compute_sinr_per_UT, compute_rate
# Tentukan direktori data
from config import SATELLITE_LIST, R_MIN, UT_COUNT

data_root = os.path.join(local_path, "../data/processed")
result_dir = os.path.join(local_path, "../results/baseline_results")
os.makedirs(result_dir, exist_ok=True)

# =============================
# Fungsi Testing Baseline
# =============================
def test_equal_power_baseline(M_val, csv_path, output_csv):
    df = pd.read_csv(csv_path)
    results = []

    with torch.no_grad():
        for sample_id, df_sample in tqdm(df.groupby("sample_id"), desc=f"Testing Baseline M={M_val}"):
            path_loss_db = df_sample["path_loss_db"].values.reshape(M_val, K)
            path_loss_tensor = torch.tensor(path_loss_db, dtype=torch.float32, device=device).unsqueeze(0)

            # ==== Bagi rata daya: Tiap satelit alokasikan P_MAX, dibagi ke K UT ====
            equal_alloc = torch.full((1, M_val, K), P_MAX / K, device=device)  # Alokasikan daya merata
            equal_alloc_scaled = equal_alloc * P_MAX  # Skala daya sesuai dengan P_MAX

            # ==== Komputasi SINR, Rate, QoS ====
            sigma_n2 = compute_noise_power()
            v_mk, _, h_mk = compute_beamforming(equal_alloc_scaled, path_loss_tensor)
            sinr_k = compute_sinr_per_UT(v_mk, h_mk, sigma_n2)
            rate_k = compute_rate(sinr_k)

            Ik = (rate_k >= R_MIN).float()  # QoS: 1 jika rate >= R_MIN, 0 jika tidak
            p_total = equal_alloc.sum(dim=2)  # Total power yang dialokasikan per satelit

            # Menghitung hasil per UT
            rates = rate_k[0].cpu().numpy().tolist()
            qos_count = int(Ik[0].sum().item())  # Menghitung jumlah UT yang memenuhi QoS
            total_rate = sum(rates)  # Total rate
            power_sat = p_total[0].cpu().numpy().tolist()  # Daya yang dialokasikan per satelit
            alloc_sat_ut = equal_alloc[0].cpu().numpy().tolist()  # Alokasi daya per satelit dan UT

            # Menyimpan hasil pengujian
            results.append({
                "sample_id": sample_id,
                "qos_count": qos_count,
                "rate_total": total_rate,
                "rate_per_ut": json.dumps(rates),
                "power_per_sat": json.dumps(power_sat),
                "alloc_per_sat": json.dumps(alloc_sat_ut)
            })

    # Menyimpan hasil ke file CSV
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)
    print(f"[DONE] Hasil testing disimpan di: {output_csv}")

# =============================
# Loop untuk semua nilai M
# =============================
def run(n_sats_list, data_dir, result_dir, ut_count, r_min):
    for M_val in n_sats_list:
        print(f"📊 Menjalankan pengujian baseline untuk {M_val} satelit...")
        
        test_path = os.path.join(data_dir, f"test_data_{M_val}sat.csv")
        output_csv = os.path.join(result_dir, f"test_result_{M_val}sat.csv")
        
        test_equal_power_baseline(M_val, test_path, output_csv)
