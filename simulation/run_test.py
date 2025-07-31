import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import json


# =============================
# Konfigurasi & Hyperparam
# =============================
# K = 5  # Jumlah User Terminals (UT)
# P_MAX = 300.0  # Daya maksimum yang diizinkan
# R_MIN = 0.35 # Threshold untuk QoS (Rate minimum per UT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils import compute_noise_power, compute_beamforming, compute_sinr_per_UT, compute_rate
from config import SATELLITE_LIST, UT_COUNT, K, P_MAX, R_MIN
# Path data dan model
parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(parent_dir)  # Naik satu level ke direktori induk

# Definisikan path baru untuk data, model, dan hasil yang berada di luar folder proyek
data_root = os.path.join(parent_dir, "data/processed")
model_root = os.path.join(parent_dir, "models")
result_dir = os.path.join(parent_dir, "results/model_results")
os.makedirs(result_dir, exist_ok=True)

# =============================
# Definisikan Model Langsung Di Sini
# =============================
class PowerAllocatorModel_1_output(torch.nn.Module):
    def __init__(self, M, K):
        super().__init__()
        self.M = M
        self.K = K
        input_dim = M * K  # Input dimensi berdasarkan jumlah satelit dan UT

        # Layer belakang (backbone)
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.LayerNorm(128),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Linear(128, 64),
            torch.nn.LayerNorm(64),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Linear(64, 32),
            torch.nn.LayerNorm(32),
            torch.nn.LeakyReLU(0.2),
        )

        # Output head untuk alokasi daya [M*K]
        self.head_power = torch.nn.Sequential(
            torch.nn.Linear(32, M * K),
            torch.nn.Softplus()  # Memastikan outputnya positif
        )

    def forward(self, x):
        h = self.backbone(x)
        power_out = self.head_power(h).view(-1, self.M, self.K)  # Output [B, M, K]
        return power_out


# =============================
# Fungsi Testing
# =============================
def test_model_csv_based(M_val, csv_path, model_path, output_csv):
    model = PowerAllocatorModel_1_output(M_val, K).to(device)  # Membuat model dengan jumlah satelit M_val
    model.load_state_dict(torch.load(model_path, map_location=device))  # Memuat model terlatih
    model.eval()  # Set model ke mode evaluasi

    df = pd.read_csv(csv_path)
    results = []

    with torch.no_grad():
        for sample_id, df_sample in tqdm(df.groupby("sample_id"), desc=f"Testing M={M_val}"):
            path_loss_db = df_sample["path_loss_db"].values.reshape(M_val, K)
            path_loss_tensor = torch.tensor(path_loss_db, dtype=torch.float32, device=device).unsqueeze(0)

            pl_linear = 10 ** (-path_loss_tensor / 10)
            x_flat = pl_linear.view(1, -1)

            predicted_power = model(x_flat)
            predicted_power_scaled = predicted_power * P_MAX

            # Menggunakan fungsi dari utils.py
            sigma_n2 = compute_noise_power()
            v_mk, _, h_mk = compute_beamforming(predicted_power_scaled, path_loss_tensor)
            sinr_k = compute_sinr_per_UT(v_mk, h_mk, sigma_n2)
            rate_k = compute_rate(sinr_k)

            Ik = (rate_k >= R_MIN).float()
            p_total = predicted_power.sum(dim=2)

            rates = rate_k[0].cpu().numpy().tolist()
            qos_count = int(Ik[0].sum().item())
            total_rate = sum(rates)
            power_sat = p_total[0].cpu().numpy().tolist()
            alloc_sat_ut = predicted_power[0].cpu().numpy().tolist()

            results.append({
                "sample_id": sample_id,
                "qos_count": qos_count,
                "rate_total": total_rate,
                "rate_per_ut": json.dumps(rates),
                "power_per_sat": json.dumps(power_sat),
                "alloc_per_sat": json.dumps(alloc_sat_ut)
            })

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)
    print(f"[DONE] Hasil testing disimpan di: {output_csv}")


# =============================
# Loop untuk semua nilai M
# =============================
def run_testing(n_sats_list, data_dir, model_dir, result_dir, ut_count, r_min):
    # Ganti 'n_sats_list' dengan 'SATELLITE_LIST' dari config.py
    for M_val in n_sats_list:
        print(f"📊 Menjalankan pengujian untuk {M_val} satelit...")

        # Path model dan data
        test_path = os.path.join(data_dir, f"test_data_{M_val}sat.csv")
        model_path = os.path.join(model_dir, f"best_model_{M_val}sat.pt") 
        output_csv = os.path.join(result_dir, f"test_result_{M_val}sat.csv")

        # Panggil fungsi testing
        test_model_csv_based(M_val, test_path, model_path, output_csv)


# Menjalankan pengujian
if __name__ == "__main__":
    run_testing()
