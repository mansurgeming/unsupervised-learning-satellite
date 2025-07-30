import os
import sys
import torch
import torch.nn as nn
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

# ================================================
# DEFINISI FUNGSI PEMBANTU
# Sebaiknya, pindahkan ini ke file 'src/utils.py' dan impor dari sana
# untuk menghindari duplikasi kode antara run_test.py dan run_baseline.py
# ================================================
def compute_noise_power(N0_dBm=-174, nf_db=7, bw_hz=40e6):
    """Menghitung daya noise dalam satuan linear (Watt)."""
    noise_dbm = N0_dBm + nf_db + 10 * np.log10(bw_hz)
    return 10**(((noise_dbm) - 30) / 10)

def compute_beamforming(p, pl_db):
    """Menghasilkan channel dan beamforming vector."""
    l=10**(-pl_db/10);p_comp=p/(l+1e-12);s_p=torch.sqrt(p_comp);k=p.shape[2];b=(k/(k+1))*l;ld=(1/(k+1))*l;phi=torch.rand_like(l)*2*np.pi-np.pi;los=torch.sqrt(b)*torch.exp(1j*phi);re=torch.randn_like(l)*torch.sqrt(ld/2);im=torch.randn_like(l)*torch.sqrt(ld/2);h=torch.sqrt(l)+(re+1j*im);v=s_p*h;return v,s_p,h

def compute_sinr_per_ut(v, h, sigma_n2):
    """Menghitung SINR per UT."""
    b,m,k=v.shape;s_k=[]
    for i in range(k):
        h_k=h[:,:,i];vH_hk=torch.einsum("bmk,bm->bk",v.conj(),h_k);i_all=torch.abs(vH_hk)**2;sig=i_all[:,i];inter=i_all.sum(dim=1)-sig;s_k.append(sig/(inter+sigma_n2));
    return torch.stack(s_k,dim=1)

def compute_rate(sinr):
    """Menghitung data rate."""
    return (240/300)*torch.log2(1+sinr)

# ================================================
# FUNGSI UTAMA (MAIN FUNCTION)
# ================================================

def run(n_sats_list, data_dir, result_dir, ut_count=5, p_max=300.0, r_min=0.5):
    """
    Fungsi utama untuk menjalankan simulasi baseline (equal power).
    Membaca data dari `data_dir` dan menyimpan hasil ke `result_dir`.
    """
    print(f"--- Memulai Proses Pengujian Baseline (Equal Power) ---")
    os.makedirs(result_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    for m_val in n_sats_list:
        test_path = os.path.join(data_dir, f"test_data_{m_val}sat.csv")
        output_csv = os.path.join(result_dir, f"test_result_{m_val}sat.csv")
        
        if not os.path.exists(test_path):
            print(f"⚠️  [PERINGATAN] File data uji tidak ditemukan: {test_path}. Melewati..."); continue

        print(f"\n[INFO] Menguji baseline untuk {m_val} satelit...")
        
        df = pd.read_csv(test_path)
        results = []

        with torch.no_grad():
            for sample_id, df_sample in tqdm(df.groupby("sample_id"), desc=f"Baseline M={m_val}", file=sys.stdout):
                path_loss_db = df_sample["path_loss_db"].values.reshape(m_val, ut_count)
                path_loss_tensor = torch.tensor(path_loss_db, dtype=torch.float32, device=device).unsqueeze(0)

                # ==== Logika Baseline: Alokasi daya dibagi rata ====
                equal_alloc_scaled = torch.full((1, m_val, ut_count), p_max / ut_count, device=device)

                # ==== Komputasi SINR, Rate, QoS ====
                sigma_n2 = compute_noise_power()
                v_mk, _, h_mk = compute_beamforming(equal_alloc_scaled, path_loss_tensor)
                sinr_k = compute_sinr_per_ut(v_mk, h_mk, sigma_n2)
                rate_k = compute_rate(sinr_k)

                ik = (rate_k >= r_min).float()

                results.append({
                    "sample_id": sample_id,
                    "qos_count": int(ik.sum().item()),
                    "rate_total": float(rate_k.sum().item()),
                    "rate_per_ut": json.dumps(rate_k[0].cpu().numpy().tolist()),
                    "power_per_sat": json.dumps(equal_alloc_scaled.sum(dim=2)[0].cpu().numpy().tolist()),
                    "alloc_per_sat": json.dumps(equal_alloc_scaled[0].cpu().numpy().tolist())
                })

        df_out = pd.DataFrame(results)
        df_out.to_csv(output_csv, index=False)
        print(f"✅ [SELESAI] Hasil baseline disimpan di: {output_csv}")
    print("\n--- Semua Proses Pengujian Baseline Selesai ---")

# ================================================
# BLOK EKSEKUSI (JIKA FILE DIJALANKAN LANGSUNG)
# ================================================

if __name__ == "__main__":
    print("Menjalankan run_baseline.py sebagai skrip mandiri...")

    # Contoh konfigurasi untuk pengujian
    N_SATS_TO_TEST = [1, 2]
    DATA_DIR_TEST = "data/processed"
    RESULT_DIR_TEST = "results/baseline_results"

    # Memanggil fungsi utama
    run(
        n_sats_list=N_SATS_TO_TEST,
        data_dir=DATA_DIR_TEST,
        result_dir=RESULT_DIR_TEST
    )