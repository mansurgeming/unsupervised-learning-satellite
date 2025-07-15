import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import csv
1
# ==== Config & Hyperparams ====
ALPHA = 0.8
BETA = 0.1
GAMMA = 0.1
ETA = 0.001
R_MIN = 1
P_MAX = 800.0
EPOCHS = 50
BATCH_SIZE = 1
K = 10 # Jumlah User Terminals (UT)

# define log file
log_file = "training_log.csv"
fieldnames = (
    ["epoch", "loss", "total_power", "sample_ids"]
    + [f"vk_mag_ut_{i+1}"  for i in range(K)]
    + [f"vk_real_ut_{i+1}" for i in range(K)]
    + [f"vk_imag_ut_{i+1}" for i in range(K)]
    + [f"sinr_ut_{i+1}" for i in range(K)]
    + [f"power_ut_{i+1}" for i in range(K)]
    + [f"rate_ut_{i+1}" for i in range(K)]
    + [f"qos_ut_{i+1}" for i in range(K)]
)

# Buat header log CSV
with open(log_file, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

class PowerDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        examples = []
        for sid, group in df.groupby("sample_id"):
            # hanya path_loss_db
            pl = group[["path_loss_db"]].values    # (K,1)
            if pl.shape[0] != K:
                continue
            examples.append((pl.astype(np.float32).flatten(), int(sid)))
        feats_list, sids_list = zip(*examples)
        self.X    = torch.from_numpy(np.stack(feats_list, axis=0))  # (N, K)
        self.sids = list(sids_list)
    def __len__(self):   return len(self.sids)
    def __getitem__(self, i): return self.X[i], self.sids[i]

class PowerAllocatorModel(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K+1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, K),
            nn.Softplus()   # agar output ≥ 0
        )
    def forward(self, x):
        # x: (B, K)
        return self.net(x)  # → (B, K)

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
                           shuffle=True)

model = PowerAllocatorModel(K)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    log_data   = {}

    for batch_idx, (path_loss_db, sample_ids) in enumerate(train_loader):
        df_all = pd.read_csv("data/processed/train_data.csv")
        PL_MAX = df_all["path_loss_db"].max()
        # ===== buat input model: [path_loss_db | P_MAX] → (B, K+1)
        B = path_loss_db.size(0)
        pl_norm   = path_loss_db / PL_MAX
        pmax_norm = torch.ones((B,1), device=path_loss_db.device)
        inp       = torch.cat([pl_norm, pmax_norm], dim=1)

        # ===== forward & hitung loss =====
        predicted_power = model(inp)                       # → (B, K)
        predicted_power = predicted_power * P_MAX

        # beamforming / SINR / rate / QoS dst, tapi path_loss_db=ganti
        sigma_n2_val = compute_noise_power()
        v_k, sqrt_p, h_mk    = compute_beamforming(predicted_power, path_loss_db)
        sinr_k               = compute_sinr(v_k, h_mk, sigma_n2_val)
        Rk                   = compute_rate(sinr_k)
        Ik                   = determine_qos(Rk)
        pk                   = aggregate_power(predicted_power)
        loss                 = custom_loss(Rk, Ik, pk)

        vk0 = v_k[0]

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

        # # Debug print komponen loss
        # if batch_idx == 0:
        #     print(f"  L1(reward_throughput) = {L1.item():.4f}")
        #     print(f"  L2(reward_qos)        = {L2.item():.4f}")
        #     print(f"  L3(qos_penalty)       = {L3.item():.4f}")
        #     print(f"  L4(power_penalty)     = {L4.item():.4f}")
        #     print(f"  L5(reg_power)         = {L5.item():.4f}")
        #     print(f"  total loss            = {lossHHH.item():.4f}")


        # # 2) Debug print gradien norm
        # if batch_idx == 0:
        #     total_norm = 0.0
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
        #             param_norm = param.grad.data.norm(2).item()
        #             total_norm += param_norm**2
        #             print(f"   grad_norm {name}: {param_norm:.4e}")
        #     total_norm = total_norm**0.5
        #     print(f"   ==> total grad norm: {total_norm:.4e}")

        if batch_idx == 0:
            log_data["epoch"] = epoch + 1
            log_data["loss"]  = loss.item()
            log_data["total_power"] = pk[0].item()
            log_data["sample_ids"]  = ", ".join(map(str, sample_ids))

            # Debug: cek shape
            print(f"\n[Epoch {epoch+1}]")
            print(f"predicted_power.shape = {predicted_power.shape}")

            # Print power per UT
            print("Power allocation per UT:")
            for i in range(K):
                # ambil nilai power
                power_val = predicted_power[0, i].item()
                print(f"  UT-{i+1}: {power_val:.6f} W")

                # simpan juga ke log_data
                log_data[f"power_ut_{i+1}"] = power_val

                # sisanya tetap seperti semula
                log_data[f"vk_mag_ut_{i+1}"]  = vk0[i].abs().item()
                log_data[f"vk_real_ut_{i+1}"] = vk0[i].real.item()
                log_data[f"vk_imag_ut_{i+1}"] = vk0[i].imag.item()
                log_data[f"sinr_ut_{i+1}"]     = sinr_k[0,i].item()
                log_data[f"rate_ut_{i+1}"]     = Rk[0,i].item()
                log_data[f"qos_ut_{i+1}"]      = Ik[0,i].item()

            print(f"Total power: {pk[0].item():.6f} W\n")

    # Simpan ke CSV
    with open(log_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(log_data)

    print(f"Epoch {epoch+1}: Loss = {total_loss:.8f}")