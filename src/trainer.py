import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import csv
import sys
import matplotlib.pyplot as plt

# =================================================================
# ==== Config & Hyperparams ====
# =================================================================
ALPHA = 0.5
BETA = 10
GAMMA = 10
ETA = 10
R_MIN = 1.0
P_MAX = 800.0
EPOCHS = 30
BATCH_SIZE = 32
K = 10
LEARNING_RATE = 1e-4

# =================================================================
# ==== Setup Logging ====
# =================================================================
log_file = "training_log_revised.csv"
fieldnames = (
    ["epoch", "loss", "total_power_watt", "sample_ids"]
    + [f"vk_mag_ut_{i+1}"  for i in range(K)]
    + [f"vk_real_ut_{i+1}" for i in range(K)]
    + [f"vk_imag_ut_{i+1}" for i in range(K)]
    + [f"sinr_ut_{i+1}" for i in range(K)]
    + [f"power_ut_{i+1}" for i in range(K)]
    + [f"rate_ut_{i+1}" for i in range(K)]
    + [f"qos_ut_{i+1}" for i in range(K)]
)

with open(log_file, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

# =================================================================
# ==== Dataset Class ====
# =================================================================
class PowerDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        examples = []
        for sid, group in df.groupby("sample_id"):
            pl = group[["path_loss_db"]].values
            if pl.shape[0] != K:
                continue
            examples.append((pl.astype(np.float32).flatten(), int(sid)))
        feats_list, sids_list = zip(*examples)
        self.X    = torch.from_numpy(np.stack(feats_list, axis=0))
        self.sids = list(sids_list)
    def __len__(self):  return len(self.sids)
    def __getitem__(self, i): return self.X[i], self.sids[i]

# =================================================================
# ==== Arsitektur Model ====
# =================================================================
class PowerAllocatorModel(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K + 1, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, K),
            nn.Softplus()
        )

    def forward(self, x):
        return self.net(x)

# =================================================================
# ==== Loss Function (REVISI PENTING) ====
# =================================================================
def custom_loss(Rk, Ik, pk_norm, alpha=ALPHA, beta=BETA, gamma=GAMMA, eta=ETA, R_min=R_MIN):
    reward_throughput = -(1 - alpha) * torch.sum(Rk * Ik)
    reward_qos = -alpha * torch.sum(Ik)
    qos_penalty = beta * torch.sum(torch.relu(R_min * Ik - Rk))
    
    # REVISI: Hapus .sum(dim=1) yang berlebihan
    # OLD: power_penalty = gamma * torch.sum(torch.relu(pk_norm.sum(dim=1) - 1.0))
    power_penalty = gamma * torch.sum(torch.relu(pk_norm - 1.0))

    reg_power = eta * pk_norm.sum()
    return reward_throughput + reward_qos + qos_penalty + power_penalty + reg_power

# =================================================================
# ==== Fungsi Pembantu ====
# =================================================================
def compute_noise_power(N0_dBm=-174, noise_figure_dB=7, bandwidth_Hz=20e6):
    total_noise_dBm = N0_dBm + noise_figure_dB + 10 * np.log10(bandwidth_Hz)
    sigma_n2 = 10 ** ((total_noise_dBm - 30) / 10)
    return sigma_n2

def compute_beamforming(predicted_power, path_loss_db):
    sqrt_p = torch.sqrt(predicted_power + 1e-9)
    L_mk = 10 ** (-path_loss_db / 10)
    beta_mk = (K / (K + 1)) * L_mk
    lambda_mk = (1.0 / (K + 1)) * L_mk
    phi_mk = torch.rand_like(L_mk) * 2 * np.pi - np.pi
    los = torch.sqrt(beta_mk) * torch.exp(1j * phi_mk)
    real = torch.randn_like(L_mk) * torch.sqrt(lambda_mk / 2)
    imag = torch.randn_like(L_mk) * torch.sqrt(lambda_mk / 2)
    nlos = real + 1j * imag
    h_mk = los + nlos
    v_k = sqrt_p * h_mk
    return v_k, sqrt_p, h_mk

def compute_sinr(v_k, h_mk, sigma_n2):
    numerator = torch.abs(v_k * h_mk) ** 2
    total_signal = torch.sum(torch.abs(v_k * h_mk) ** 2, dim=1, keepdim=True)
    denominator = total_signal - numerator + sigma_n2 + 1e-9
    return numerator / denominator

def compute_rate(sinr_k, tau_d=270, tau_c=300):
    return (tau_d / tau_c) * torch.log2(1 + sinr_k)

def determine_qos(Rk, R_min=R_MIN, Ik):
    return torch.sigmoid(Ik * (Rk - R_min))

def aggregate_power(predicted_power):
    return predicted_power.sum(dim=1)
# =================================================================
# ==== Fungsi Plotting (FUNGSI BARU) ====
# =================================================================
def plot_training_results(log_path):
    """Membaca file log CSV dan membuat grafik loss."""
    print("\nMembuat grafik hasil training...")
    try:
        df = pd.read_csv(log_path)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plotting Loss dengan skala logaritmik
        ax.plot(df['epoch'], df['loss'], marker='o', linestyle='-', color='b', label='Loss per Batch (Skala Log)')
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (Skala Log)")
        ax.set_title("Grafik Penurunan Loss per Epoch")
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--")

        plt.tight_layout()
        
        # Menyimpan plot ke file
        output_filename = "loss_vs_epoch.png"
        plt.savefig(output_filename)
        print(f"Grafik telah berhasil disimpan sebagai: {output_filename}")

    except FileNotFoundError:
        print(f"Error: File log '{log_path}' tidak ditemukan.")
    except Exception as e:
        print(f"Error saat membuat grafik: {e}")
# =================================================================
# ==== Persiapan & Training Loop ====
# =================================================================
train_dataset = PowerDataset("data/processed/train_data.csv")
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = PowerAllocatorModel(K)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Membaca data untuk normalisasi...")
df_all = pd.read_csv("data/processed/train_data.csv")
PL_MAX = df_all["path_loss_db"].max()
print(f"Selesai. PL_MAX ditemukan: {PL_MAX}")

for epoch in range(EPOCHS):
    model.train()
    total_epoch_loss = 0
    
    for batch_idx, (path_loss_db, sample_ids) in enumerate(train_loader):
        B = path_loss_db.size(0)
        if B <= 1:
            continue
            
        pl_norm   = path_loss_db / PL_MAX
        pmax_norm = torch.ones((B,1), device=path_loss_db.device)
        inp       = torch.cat([pl_norm, pmax_norm], dim=1)

        predicted_power_norm = model(inp)
        predicted_power_scaled = predicted_power_norm * P_MAX

        sigma_n2_val = compute_noise_power()
        v_k, _, h_mk = compute_beamforming(predicted_power_scaled, path_loss_db)
        sinr_k = compute_sinr(v_k, h_mk, sigma_n2_val)
        Rk = compute_rate(sinr_k)
        Ik = determine_qos(Rk)
        
        pk_scaled = aggregate_power(predicted_power_scaled)
        pk_norm = aggregate_power(predicted_power_norm)

        loss = custom_loss(Rk, Ik, pk_norm)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_epoch_loss += loss.item()

        if batch_idx == 0:
            log_data = {}
            log_data["epoch"] = epoch + 1
            log_data["loss"]  = loss.item()
            log_data["total_power_watt"] = pk_scaled[0].item()
            log_data["sample_ids"] = str(sample_ids[0].item())
            
            for i in range(K):
                log_data[f"power_ut_{i+1}"] = predicted_power_scaled[0, i].item()
                log_data[f"vk_mag_ut_{i+1}"]  = v_k[0,i].abs().item()
                log_data[f"vk_real_ut_{i+1}"] = v_k[0,i].real.item()
                log_data[f"vk_imag_ut_{i+1}"] = v_k[0,i].imag.item()
                log_data[f"sinr_ut_{i+1}"]    = sinr_k[0,i].item()
                log_data[f"rate_ut_{i+1}"]    = Rk[0,i].item()
                log_data[f"qos_ut_{i+1}"]     = Ik[0,i].item()

            with open(log_file, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(log_data)
    
    # Hitung rata-rata loss per epoch
    if len(train_loader) > 0:
        avg_loss = total_epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}: Average Loss = {avg_loss:.6f}")
        sys.stdout.flush()

# =================================================================
# ==== Panggil Fungsi Plotting Setelah Training Selesai ====
# =================================================================
plot_training_results(log_file)