import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import csv
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

# =================================================================
# ==== Config & Hyperparams ====
# =================================================================
NUM_SAT = 4  # Ganti angka ini (1, 2, 4, 6, 8)
K = 10       # Jumlah UT (User Terminals)
M = NUM_SAT  # Jumlah Satelit

ALPHA = 0.5
BETA = 5
GAMMA = 5
R_MIN = 1.5
P_MAX = 800.0
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# =================================================================
# ==== Setup Logging & Direktori Output ====
# =================================================================
output_dir = f"results/train_{NUM_SAT}satellite_per_link"
os.makedirs(output_dir, exist_ok=True)
print(f"[INFO] Semua hasil akan disimpan di folder: '{output_dir}'")

log_file = os.path.join(output_dir, f"training_log_{NUM_SAT}sat_per_link.csv")

# --- REVISI UTAMA: Mengelompokkan header berdasarkan jenis fitur ---
# 1. Tentukan kolom-kolom dasar
fieldnames = ["epoch", "loss", "total_power_watt", "sample_ids"]

# 2. Tentukan urutan fitur yang Anda inginkan
feature_prefixes = ["power", "sinr", "rate", "qos"]

# 3. Buat header dengan mengulang untuk setiap jenis fitur
for prefix in feature_prefixes:
    for i in range(K):
        for j in range(M):
            ut_id, sat_id = i + 1, j + 1
            # Tambahkan kolom sesuai format: feature_utX_satY
            fieldnames.append(f"{prefix}_ut{ut_id}_sat{sat_id}")

with open(log_file, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

# =================================================================
# ==== Dataset Class (Direvisi untuk Input Semua Link) ====
# =================================================================
class PowerDataset(Dataset):
    def __init__(self, path, num_sat, num_ut):
        df = pd.read_csv(path)
        
        # Ambil semua kolom path loss
        pl_cols = [f"path_loss_db_sat_{i+1}" for i in range(num_sat)]
        
        examples = []
        for sid, group in tqdm(df.groupby("sample_id"), desc="Preprocessing All Links"):
            if len(group) == num_ut:
                # Susun path loss menjadi matriks (UTs x SATs)
                sample_features = group[pl_cols].values.astype(np.float32)
                # Ratakan (flatten) matriks menjadi vektor panjang
                examples.append((sample_features.flatten(), int(sid)))

        if not examples:
            raise ValueError("Tidak ada data valid yang bisa diproses.")

        feats_list, sids_list = zip(*examples)
        self.X = torch.from_numpy(np.stack(feats_list, axis=0))
        self.sids = list(sids_list)

    def __len__(self): return len(self.sids)
    def __getitem__(self, i): return self.X[i], self.sids[i]

# =================================================================
# ==== Arsitektur Model (Input & Output Disesuaikan) ====
# =================================================================
class PowerAllocatorModel(nn.Module):
    def __init__(self, K, M):
        super().__init__()
        # --- PERUBAHAN: Ukuran input dan output disesuaikan ---
        num_inputs = K * M + 1 # (10 UT * 4 SAT) + 1 untuk P_MAX
        num_outputs = K * M
        
        self.net = nn.Sequential(
            nn.Linear(num_inputs, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Linear(64, num_outputs), nn.Softplus()
        )
    def forward(self, x): return self.net(x)

# =================================================================
# ==== Loss Function & Fungsi Pembantu ====
# =================================================================
def custom_loss(Rk, Ik, power_per_satellite_norm, alpha=ALPHA, beta=BETA, gamma=GAMMA, R_min=R_MIN):
    # Rk dan Ik sekarang berbentuk [Batch, K, M]
    # pk_norm sekarang adalah total daya dari semua K*M link
    reward_throughput = -(1 - alpha) * torch.sum(Rk * Ik)
    reward_qos = -alpha * torch.sum(Ik)
    qos_penalty = beta * torch.sum(torch.relu(R_min * Ik - Rk))
    # power_penalty = gamma * torch.sum(torch.relu(pk_norm - 1.0))
    power_penalty = gamma * torch.sum(torch.relu(power_per_satellite_norm - 1.0))
    return reward_throughput + reward_qos + qos_penalty + power_penalty

def compute_noise_power(N0_dBm=-174, noise_figure_dB=7, bandwidth_Hz=20e6):
    total_noise_dBm = N0_dBm + noise_figure_dB + 10 * np.log10(bandwidth_Hz)
    return 10 ** ((total_noise_dBm - 30) / 10)

def compute_beamforming(predicted_power, path_loss_db):
    # predicted_power dan path_loss_db sekarang berbentuk [Batch, K, M]
    L_mk = 10 ** (-path_loss_db / 10)
    p = predicted_power / L_mk
    sqrt_p = torch.sqrt(p)
    # Perhitungan channel tetap sama, dilakukan per elemen
    beta_mk = (K / (K + 1)) * L_mk # Ini asumsi, bisa disesuaikan
    lambda_mk = (1.0 / (K + 1)) * L_mk
    phi_mk = torch.rand_like(L_mk) * 2 * np.pi - np.pi
    los = torch.sqrt(beta_mk) * torch.exp(1j * phi_mk)
    real = torch.randn_like(L_mk) * torch.sqrt(lambda_mk / 2)
    imag = torch.randn_like(L_mk) * torch.sqrt(lambda_mk / 2)
    nlos = real + 1j * imag
    h_mk = torch.sqrt(L_mk) + nlos
    v_k = sqrt_p * h_mk
    return v_k, sqrt_p, h_mk

def compute_sinr(v_k, h_mk, sigma_n2):
    # v_k dan h_mk sekarang berbentuk [Batch, K, M]
    numerator = torch.abs(v_k * h_mk) ** 2
    # --- PERUBAHAN: Penjumlahan total sinyal pengganggu dari SEMUA link ---
    # total_signal = torch.sum(torch.abs(numerator), dim=[1, 2], keepdim=True)
    total_signal = torch.sum(numerator, dim=[1, 2], keepdim=True)
    denominator = total_signal - numerator + sigma_n2
    return numerator / denominator

def compute_rate(sinr_k, tau_d=270, tau_c=300):
    return (tau_d / tau_c) * torch.log2(1 + 100 * sinr_k)

# def aggregate_power(predicted_power):
#     # --- PERUBAHAN: Menjumlahkan daya dari semua K*M link ---
#     return predicted_power.sum(dim=1)
def aggregate_power_per_satellite(predicted_power):
    # predicted_power berbentuk [Batch, K, M]
    # Kita ingin menjumlahkan daya sepanjang dimensi UT (dim=1)
    # Hasilnya akan berbentuk [Batch, M], yaitu total daya per satelit.
    return predicted_power.sum(dim=1)

# =================================================================
# ==== FUNGSI PLOTTING (BARU & DIREVISI) ====
# =================================================================
def plot_training_loss(log_path, save_dir):
    print("\nMembuat grafik hasil training (Loss)...")
    try:
        df = pd.read_csv(log_path)
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(df['epoch'], df['loss'], marker='o', linestyle='-', color='b', label='Loss per Batch')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Value")
        ax.set_title("Grafik Penurunan Loss per Epoch")
        ax.legend()
        ax.grid(True, which="both", ls="--")
        plt.tight_layout()
        output_filename = os.path.join(save_dir, "loss_vs_epoch.png")
        plt.savefig(output_filename)
        print(f"Grafik loss telah disimpan sebagai: {output_filename}")
        plt.close()
    except Exception as e:
        print(f"Error saat membuat grafik loss: {e}")

def plot_per_link_metrics(log_path, save_dir, metric_prefix, y_label):
    """Fungsi generik untuk membuat plot rate dan QoS per link."""
    print(f"\nMembuat grafik {y_label} vs Epoch untuk setiap link...")
    try:
        df = pd.read_csv(log_path)
        # Buat subfolder untuk menyimpan plot
        plot_subdir = os.path.join(save_dir, f"{metric_prefix}_per_link_plots")
        os.makedirs(plot_subdir, exist_ok=True)

        # Buat satu plot untuk setiap UT
        for i in range(1, K + 1):
            plt.figure(figsize=(12, 7))
            
            # Di setiap plot, gambar satu garis untuk setiap Satelit
            for j in range(1, M + 1):
                col_name = f"{metric_prefix}_ut{i}_sat{j}"
                if col_name in df.columns:
                    plt.plot(df["epoch"], df[col_name], marker='o', linestyle='-', markersize=4, label=f'Link to SAT {j}')
            
            plt.xlabel("Epoch")
            plt.ylabel(y_label)
            plt.title(f"{y_label} vs. Epoch for UT {i}")
            plt.grid(True)
            plt.legend()
            
            # Atur batas y-axis khusus untuk plot QoS
            if metric_prefix == 'qos':
                plt.ylim(-0.1, 1.1)

            output_file = os.path.join(plot_subdir, f"{metric_prefix}_vs_epoch_ut_{i}.png")
            plt.savefig(output_file)
            plt.close()
            
        print(f"Grafik {y_label} per link telah disimpan di folder '{plot_subdir}'.")

    except Exception as e:
        print(f"Error saat membuat grafik {y_label} per link: {e}")


# =================================================================
# ==== Persiapan & Training Loop ====
# =================================================================
train_file_path = f"data/processed/train_data_wide_{NUM_SAT}sat.csv"
if not os.path.exists(train_file_path):
    print(f"[ERROR] File data tidak ditemukan: {train_file_path}"); sys.exit()

print(f"Memuat dataset dari: {train_file_path}")
train_dataset = PowerDataset(train_file_path, num_sat=M, num_ut=K)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- PERUBAHAN: Inisialisasi model dengan K dan M ---
model = PowerAllocatorModel(K, M)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Membaca data untuk normalisasi...")
df_all = pd.read_csv(train_file_path)
pl_cols_all = [f"path_loss_db_sat_{i+1}" for i in range(M)]
PL_MAX = df_all[pl_cols_all].max().max()
print(f"Selesai. PL_MAX ditemukan: {PL_MAX}")

for epoch in range(EPOCHS):
    model.train()
    total_epoch_loss = 0
    for batch_idx, (path_loss_flat, sample_ids) in enumerate(train_loader):
        B = path_loss_flat.size(0)
        if B <= 1: continue
        
        # --- PERUBAHAN: Siapkan input untuk model ---
        pl_norm_flat = path_loss_flat / PL_MAX
        pmax_norm = torch.ones((B, 1), device=path_loss_flat.device)
        inp = torch.cat([pl_norm_flat, pmax_norm], dim=1)
        
        predicted_power_norm_flat = model(inp)
        
        # --- PERUBAHAN: Reshape semua tensor ke [Batch, UT, SAT] ---
        path_loss_db = path_loss_flat.view(B, K, M)
        predicted_power_norm = predicted_power_norm_flat.view(B, K, M)
        
        predicted_power_scaled = predicted_power_norm * P_MAX
        
        # Lanjutkan dengan perhitungan seperti biasa, tensor sudah dalam bentuk matriks
        sigma_n2_val = compute_noise_power()
        v_k, _, h_mk = compute_beamforming(predicted_power_scaled, path_loss_db)
        sinr_k = compute_sinr(v_k, h_mk, sigma_n2_val)
        Rk = compute_rate(sinr_k)
        Ik = (Rk >= R_MIN).float()
        
        # Agregasi daya PER SATELIT
        pk_scaled_per_sat = aggregate_power_per_satellite(predicted_power_scaled)
        pk_norm_per_sat = aggregate_power_per_satellite(predicted_power_norm)
        pk_norm_per_sat = torch.relu(pk_norm_per_sat)

        # Gunakan pk_norm_per_sat di loss function
        loss = custom_loss(Rk, Ik, pk_norm_per_sat)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_epoch_loss += loss.item()

        # --- PERUBAHAN: Logging untuk setiap link ---
        if batch_idx == 0:
            # Hitung total daya sistem hanya untuk logging
            total_power_for_log = pk_scaled_per_sat.sum().item()
            
            log_data = {"epoch": epoch + 1, "loss": loss.item(), 
                        "total_power_watt": total_power_for_log, # Ini adalah total dari semua satelit
                        "sample_ids": str(sample_ids[0].item())}
            
            for i in range(K):
                for j in range(M):
                    ut_id, sat_id = i + 1, j + 1
                    log_data[f"power_ut{ut_id}_sat{sat_id}"] = predicted_power_scaled[0, i, j].item()
                    log_data[f"sinr_ut{ut_id}_sat{sat_id}"] = sinr_k[0, i, j].item()
                    log_data[f"rate_ut{ut_id}_sat{sat_id}"] = Rk[0, i, j].item()
                    log_data[f"qos_ut{ut_id}_sat{sat_id}"] = Ik[0, i, j].item()

            with open(log_file, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(log_data)
    
    if len(train_loader) > 0:
        avg_loss = total_epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}: Average Loss = {avg_loss:.6f}"); sys.stdout.flush()

print("\nTraining selesai.")
# --- PERUBAHAN: Memanggil semua fungsi plotting setelah training selesai ---
plot_training_loss(log_file, output_dir)
plot_per_link_metrics(log_file, output_dir, metric_prefix="rate", y_label="Rate (bps/Hz)")
plot_per_link_metrics(log_file, output_dir, metric_prefix="qos", y_label="QoS Indicator (1 = No Handover)")

print(f"\nSemua log dan plot disimpan di direktori: '{output_dir}'")