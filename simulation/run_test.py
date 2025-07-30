# Nama file: simulation/run_test.py

import os
import sys
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import json

# ================================================
# DEFINISI STRUKTUR MODEL & FUNGSI PEMBANTU
# ================================================
# NOTE: Kelas ini HARUS didefinisikan agar PyTorch tahu
# cara memuat bobot dari file .pt ke dalam struktur yang benar.
# Ini tidak melatih ulang model, hanya sebagai 'cetakan'.
class PowerAllocatorModel_1_output(nn.Module):
    def __init__(self, M, K):
        super().__init__()
        self.M, self.K = M, K
        input_dim = M * K
        self.backbone = nn.Sequential(nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.LeakyReLU(0.2), nn.Linear(128, 64), nn.LayerNorm(64), nn.LeakyReLU(0.2), nn.Linear(64, 32), nn.LayerNorm(32), nn.LeakyReLU(0.2))
        self.head_power = nn.Sequential(nn.Linear(32, M * K), nn.Softplus())
    def forward(self, x):
        return self.head_power(self.backbone(x)).view(-1, self.M, self.K)

# Anda bisa memindahkan fungsi-fungsi di bawah ini ke utils.py
def compute_noise_power(N0_dBm=-174, nf_db=7, bw_hz=40e6):
    return 10**(((N0_dBm + nf_db + 10 * np.log10(bw_hz)) - 30) / 10)

def compute_beamforming(p, pl_db):
    l=10**(-pl_db/10);p_comp=p/(l+1e-12);s_p=torch.sqrt(p_comp);k=p.shape[2];b=(k/(k+1))*l;ld=(1/(k+1))*l;phi=torch.rand_like(l)*2*np.pi-np.pi;los=torch.sqrt(b)*torch.exp(1j*phi);re=torch.randn_like(l)*torch.sqrt(ld/2);im=torch.randn_like(l)*torch.sqrt(ld/2);h=torch.sqrt(l)+(re+1j*im);v=s_p*h;return v,s_p,h

def compute_sinr_per_ut(v, h, sigma_n2):
    b,m,k=v.shape;s_k=[]
    for i in range(k):
        h_k=h[:,:,i];vH_hk=torch.einsum("bmk,bm->bk",v.conj(),h_k);i_all=torch.abs(vH_hk)**2;sig=i_all[:,i];inter=i_all.sum(dim=1)-sig;s_k.append(sig/(inter+sigma_n2));
    return torch.stack(s_k,dim=1)

def compute_rate(sinr):
    return (240/300)*torch.log2(1+sinr)

# ================================================
# FUNGSI UTAMA (MAIN FUNCTION)
# ================================================

def run(n_sats_list, data_dir, model_dir, result_dir, ut_count=5, p_max=300.0, r_min=0.5):
    """
    Fungsi utama untuk menjalankan pengujian pada model yang sudah dilatih.
    """
    print(f"--- Memulai Proses Pengujian Model ---")
    os.makedirs(result_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    for m_val in n_sats_list:
        test_path = os.path.join(data_dir, f"test_data_{m_val}sat.csv")
        
        # Menggunakan format nama file yang benar: best_model_numsatSatelite.pt
        model_path = os.path.join(model_dir, f"best_model_{m_val}sat.pt")
        
        output_csv = os.path.join(result_dir, f"test_result_{m_val}sat.csv")
        
        if not os.path.exists(test_path) or not os.path.exists(model_path):
            if not os.path.exists(test_path): print(f"⚠️  [PERINGATAN] File data uji tidak ditemukan: {test_path}.")
            if not os.path.exists(model_path): print(f"⚠️  [PERINGATAN] File model tidak ditemukan: {model_path}.")
            continue

        print(f"\n[INFO] Menguji model untuk {m_val} satelit...")
        
        # 1. Buat 'kerangka' model
        model = PowerAllocatorModel_1_output(m_val, ut_count).to(device)
        # 2. Muat bobot yang sudah dilatih ke dalam kerangka tersebut
        model.load_state_dict(torch.load(model_path, map_location=device))
        # 3. Atur model ke mode evaluasi (tidak ada training)
        model.eval()

        df = pd.read_csv(test_path)
        results = []

        with torch.no_grad(): # Pastikan tidak ada gradien yang dihitung
            for sample_id, df_sample in tqdm(df.groupby("sample_id"), desc=f"Testing M={m_val}", file=sys.stdout):
                # a. Ambil HANYA path_loss_db dari data
                path_loss_db = df_sample["path_loss_db"].values.reshape(m_val, ut_count)
                path_loss_tensor = torch.tensor(path_loss_db, dtype=torch.float32, device=device).unsqueeze(0)

                # b. Konversi path loss ke skala linear dan ratakan (flatten)
                pl_linear = 10 ** (-path_loss_tensor / 10)
                x_flat = pl_linear.view(1, -1)

                # c. Lakukan prediksi (inference) dengan model yang sudah di-load
                predicted_power = model(x_flat)
                predicted_power_scaled = predicted_power * p_max
                
                # d. Hitung metrik performa berdasarkan hasil prediksi
                sigma_n2 = compute_noise_power()
                v_mk, _, h_mk = compute_beamforming(predicted_power_scaled, path_loss_tensor)
                rate_k = compute_rate(compute_sinr_per_ut(v_mk, h_mk, sigma_n2))

                ik = (rate_k >= r_min).float()
                
                results.append({
                    "sample_id": int(sample_id),
                    "qos_count": int(ik.sum().item()),
                    "rate_total": float(rate_k.sum().item()),
                    "rate_per_ut": json.dumps(rate_k[0].cpu().numpy().tolist()),
                    "power_per_sat": json.dumps(predicted_power_scaled.sum(dim=2)[0].cpu().numpy().tolist()),
                    "alloc_per_sat": json.dumps(predicted_power_scaled[0].cpu().numpy().tolist())
                })

        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"✅ [SELESAI] Hasil pengujian disimpan di: {output_csv}")
    print("\n--- Semua Proses Pengujian Model Selesai ---")

if __name__ == "__main__":
    run(
        n_sats_list=[1, 2],
        data_dir="/data/processed", # Sesuaikan path relatif untuk testing
        model_dir="/models",
        result_dir="/results/model_results"
    )