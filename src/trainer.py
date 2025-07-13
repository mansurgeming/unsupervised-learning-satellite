import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# ==== Config & Hyperparams ====
ALPHA = 0.5
BETA = 10.0
GAMMA = 10.0
ETA = 10.0
R_MIN = 1.0
P_MAX = 800.0
EPOCHS = 5
BATCH_SIZE = 10
K = 10 # Jumlah User Terminals (UT)

class PowerDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        grouped = df.groupby("sample_id")
        self.X = []
        for _, group in grouped:
            features = group[["dist_km", "elev_deg", "azim_deg", "path_loss_db"]].values
            self.X.append(features)
        self.X = torch.tensor(self.X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]  # (10, 4)

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
    power_penalty = gamma * torch.relu(torch.sum(pk) - P_max)            # + gamma (Σ Pk - Pmax)
    reg_power = eta * torch.sum(pk)                                       # + eta Σ Pk
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
    beta_mk = (KAPPA / (KAPPA + 1)) * L_mk
    lambda_mk = (1.0 / (KAPPA + 1)) * L_mk

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
    inner_product = torch.sum(torch.conj(v_k) * h_mk, dim=1)
    numerator = torch.abs(inner_product) ** 2

    interference = torch.sum(torch.abs(v_k * h_mk) ** 2, dim=1)
    denominator = interference - numerator + sigma_n2
    
    return numerator / denominator

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
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = PowerAllocatorModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x in train_loader:
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

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")