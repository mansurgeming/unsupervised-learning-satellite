import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import csv

# ==== Config & Hyperparams ====
ALPHA = 0.5
BETA = 10
GAMMA = 10
ETA = 10
R_MIN = 0.000000000667
P_MAX = 800.0
EPOCHS = 5
BATCH_SIZE = 1
K = 10 # Jumlah User Terminals (UT)

# define log file
log_file = "training_log.csv"
fieldnames = (
    ["epoch", "loss", "total_power", "sample_ids"]
    + [f"power_ut_{i+1}" for i in range(K)]
    + [f"rate_ut_{i+1}" for i in range(K)]
    + [f"qos_ut_{i+1}" for i in range(K)]
    + [f"sinr_ut_{i+1}" for i in range(K)]
)

# Buat header log CSV
with open(log_file, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

class PowerDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        # buat list of (features, sample_id)
        examples = []
        for sid, group in df.groupby("sample_id"):
            feats = group[["dist_km","elev_deg","azim_deg","path_loss_db"]].values
            if feats.shape[0] != K:
                continue
            examples.append((feats.astype(np.float32), int(sid)))
        # unzip ke dua list sejajar
        feats_list, sids_list = zip(*examples)
        # jadi tensor sekali, bukan per-loop
        self.X = torch.from_numpy(np.stack(feats_list, axis=0))  # (N_samples, K, 4)
        self.sids = list(sids_list)                              # [sid0, sid1, …]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # kembalikan: (tensor-features, integer-sample_id)
        return self.X[idx], self.sids[idx]

def collate_fn(batch):
    # batch: list of tuples (features, sid)
    feats, sids = zip(*batch)
    # feats sudah tensor, tinggal stack
    feats = torch.stack(feats, dim=0)  # (B, K, 4)
    return feats, list(sids)


class PowerAllocatorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.mlp(x)

# Custom loss function sesuai persamaan:
# L = -(1 - alpha) * sum(Rk * Ik) - alpha * sum(Ik)
#     + sum(beta_k * (R_min * Ik - Rk)) + gamma * (sum(Pk) - P_max) + sum(eta_k * Pk)
def custom_loss(Rk, Ik, pk, alpha=ALPHA, beta=BETA, gamma=GAMMA, eta=ETA, R_min=R_MIN, P_max=P_MAX):
    reward_throughput = -(1 - alpha) * torch.sum(Rk * Ik)                 # - (1 - alpha) Σ Rk Ik
    reward_qos = -alpha * torch.sum(Ik)                                   # - alpha Σ Ik
    qos_penalty = beta * torch.sum(R_min * Ik - Rk)                       # + beta Σ (R_min Ik - Rk)
    power_penalty = gamma * torch.relu(pk - P_max)            # + gamma (Σ Pk - Pmax)
    reg_power = eta * pk                                       # + eta Σ Pk
    return reward_throughput + reward_qos + qos_penalty + power_penalty + reg_power

def compute_noise_power(N0_dBm=-174, noise_figure_dB=7, bandwidth_Hz=20e6):
    # Total noise power in dBm
    total_noise_dBm = N0_dBm + noise_figure_dB + 10 * np.log10(bandwidth_Hz)

    # Convert dBm to Watts: P(W) = 10^((P[dBm] - 30)/10)
    sigma_n2 = 10 ** ((total_noise_dBm - 30) / 10)
    return sigma_n2

def compute_beamforming(predicted_power, path_loss_db):
    sqrt_p = torch.sqrt(predicted_power)

    # Step 1: Hitung L_mk dari path loss
    L_mk = 10 ** (-path_loss_db / 10)

    # Step 2: Hitung beta dan lambda dari L_mk dan K-factor
    beta_mk = (K / (K + 1)) * L_mk
    lambda_mk = (1.0 / (K + 1)) * L_mk

    # Step 3: Buat LoS phase component: exp(j * phi), phi uniform [-π, π]
    phi_mk = torch.rand_like(L_mk) * 2 * np.pi - np.pi
    los = torch.sqrt(beta_mk) * torch.exp(1j * phi_mk)

    # Step 4: Buat NLoS (Rayleigh): CN(0, lambda)
    real = torch.randn_like(L_mk) * torch.sqrt(lambda_mk / 2)
    imag = torch.randn_like(L_mk) * torch.sqrt(lambda_mk / 2)
    nlos = real + 1j * imag

    # Step 5: Total channel
    h_mk = los + nlos

    # Step 6: v_k = sqrt(p_k) * h_k
    v_k = sqrt_p * h_mk

    return v_k, sqrt_p, h_mk

def compute_sinr(v_k, h_mk, sigma_n2):
    # v_k, h_mk shape: (B, K)
    numerator = torch.abs(v_k * h_mk) ** 2  # shape: (B, K)
    total_signal = torch.sum(torch.abs(v_k * h_mk) ** 2, dim=1, keepdim=True)  # (B, 1)
    denominator = total_signal - numerator + sigma_n2  # shape: (B, K)
    return numerator / denominator  # shape: (B, K)


# Rate: Rk = (tau_d / tau_c) * log2(1 + SINR_k)
def compute_rate(sinr_k, tau_d=270, tau_c=300):
    return (tau_d / tau_c) * torch.log2(1 + sinr_k)

# QoS indicator: Ik = 1 if Rk >= R_min else 0
def determine_qos(Rk, R_min=R_MIN):
    return (Rk >= R_min).float()

# Aggregate total power per UT
def aggregate_power(predicted_power):
    return predicted_power.sum(dim=1)


# ==== Prepare Data ====
train_dataset = PowerDataset("data/processed/train_data.csv")
train_loader  = DataLoader(train_dataset,
                           batch_size=BATCH_SIZE,
                           shuffle=True,
                           collate_fn=collate_fn)

model = PowerAllocatorModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    log_data   = {}

    for batch_idx, (x, sample_ids) in enumerate(train_loader):
        predicted_power = model(x)
        sigma_n2_val = compute_noise_power()
        path_loss_db = x[:, :, 3]  # kolom path_loss_db
        v_k, sqrt_p, h_mk = compute_beamforming(predicted_power, path_loss_db)

        sinr_k = compute_sinr(v_k, h_mk, sigma_n2_val)
        Rk = compute_rate(sinr_k)
        Ik = determine_qos(Rk)
        pk = aggregate_power(predicted_power)
        loss = custom_loss(Rk, Ik, pk)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        L1 = -(1-ALPHA) * torch.sum(Rk * Ik)
        L2 = -ALPHA * torch.sum(Ik)
        L3 = BETA * torch.sum(R_MIN * Ik - Rk)
        L4 = GAMMA * torch.relu(pk - P_MAX)
        L5 = ETA * pk

        # total loss
        lossHHH = L1 + L2 + L3 + L4 + L5

        # Debug print komponen loss

        if batch_idx == 0:

        #     print(f"  L1(reward_throughput) = {L1.item():.4f}")
        #     print(f"  L2(reward_qos)        = {L2.item():.4f}")
        #     print(f"  L3(qos_penalty)       = {L3.item():.4f}")
        #     print(f"  L4(power_penalty)     = {L4.item():.4f}")
        #     print(f"  L5(reg_power)         = {L5.item():.4f}")
        #     print(f"  total loss            = {lossHHH.item():.4f}")


        # 2) Debug print gradien norm
        if batch_idx == 0:
            total_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm**2
                    print(f"   grad_norm {name}: {param_norm:.4e}")
            total_norm = total_norm**0.5
            print(f"   ==> total grad norm: {total_norm:.4e}")

        if batch_idx == 0:
            log_data["epoch"] = epoch + 1
            log_data["loss"]  = loss.item()
            log_data["total_power"] = pk[0].item()
            log_data["sample_ids"]  = ", ".join(map(str, sample_ids))
            for i in range(K):
                log_data[f"power_ut_{i+1}"] = predicted_power[0,i].item()
                log_data[f"rate_ut_{i+1}"]  = Rk[0,i].item()
                log_data[f"qos_ut_{i+1}"]   = Ik[0,i].item()
                log_data[f"sinr_ut_{i+1}"]  = sinr_k[0,i].item()

    # Simpan ke CSV
    with open(log_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(log_data)

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")