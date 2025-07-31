import torch
import numpy as np
K = 5
def compute_noise_power(N0_dBm=-174, noise_figure_dB=7, bandwidth_Hz=40e6):
    """
    Hitung noise power (σ^2) berdasarkan:
    - N0: noise spectral density dalam dBm/Hz
    - noise figure (NF): tambahan noise dari sistem RF
    - bandwidth (Hz)

    Output:
        sigma_n2: noise power dalam satuan linear (Watt)
    """
    total_noise_dBm = N0_dBm + noise_figure_dB + 10 * np.log10(bandwidth_Hz)
    sigma_n2 = 10 ** ((total_noise_dBm - 30) / 10)
    return sigma_n2


def compute_beamforming(predicted_power, path_loss_db):
    """
    Generate channel dan beamforming vector berdasarkan alokasi daya dan path loss.

    Args:
        predicted_power : Tensor [M, K] → daya dari setiap satelit ke setiap UT
        path_loss_db    : Tensor [M, K] → path loss dalam dB

    Returns:
        v_mk   : beamforming vector kompleks [M, K]
        sqrt_p : akar dari daya terkompensasi path loss [M, K]
        h_mk   : channel kompleks antara satelit dan UT [M, K]
    """
    L_mk = 10 ** (-path_loss_db / 10)  # convert dB to linear scale
    p = predicted_power / L_mk  # Kompensasi path loss
    sqrt_p = torch.sqrt(p)

    beta_mk = (K / (K + 1)) * L_mk  # LoS power
    lambda_mk = (1.0 / (K + 1)) * L_mk  # NLoS power

    phi = torch.rand_like(L_mk) * 2 * np.pi - np.pi
    los = torch.sqrt(beta_mk) * torch.exp(1j * phi)

    real = torch.randn_like(L_mk) * torch.sqrt(lambda_mk / 2)
    imag = torch.randn_like(L_mk) * torch.sqrt(lambda_mk / 2)
    nlos = real + 1j * imag

    h_mk = torch.sqrt(L_mk) + nlos
    v_mk = sqrt_p * h_mk  # beamforming vector
    return v_mk, sqrt_p, h_mk


def compute_sinr_per_UT(v_mk, h_mk, sigma_n2):
    """
    Hitung SINR per UT (k) sesuai rumus:
    SINR_k = |v_k^H h_k|^2 / (sum_i |v_i^H h_k|^2 - |v_k^H h_k|^2 + sigma_n^2)

    Args:
        v_mk : Tensor [M, K] kompleks → beamforming vector
        h_mk : Tensor [M, K] kompleks → channel dari M satelit ke K UT
        sigma_n2 : float → noise power

    Returns:
        sinr_k : Tensor [K] → SINR untuk tiap UT
    """
    B, M, K = v_mk.shape
    sinr_k = []

    for k in range(K):
        h_k = h_mk[:, :, k]  # [B, M] → channel dari semua satelit ke UT-k
        v_k = v_mk[:, :, k]  # [B, M] → beam ke UT-k

        # Vectorized interference
        vH_hk = torch.einsum("bmk,bm->bk", v_mk.conj(), h_k)  # [B, K]
        interference_all = torch.abs(vH_hk) ** 2
        signal = interference_all[:, k]  # [B]
        interference = interference_all.sum(dim=1) - signal

        sinr = signal / (interference + sigma_n2)
        sinr_k.append(sinr)
    return torch.stack(sinr_k, dim=1)  # [B, K] → SINR untuk tiap UT tiap batch


def compute_rate(sinr_k, tau_d=240, tau_c=300):
    """
    Hitung data rate (bps/Hz) per UT berdasarkan SINR.

    Rumus:
        Rk = (tau_d / tau_c) * log2(1 + SINR_k)

    - sinr_k: Tensor [K], SINR per UT
    - tau_d: simbol data (default 270)
    - tau_c: total simbol (default 300)

    Output: Tensor [K], rate per UT
    """
    result = (tau_d / tau_c) * torch.log2(1 + sinr_k)
    return result


def aggregate_power_per_sat(predicted_power):
    """
    Hitung total daya per satelit.

    Args:
        predicted_power: Tensor [M, K]
            Alokasi daya dari M satelit ke K UT (per batch)

    Returns:
        Tensor [M]
            Total daya tiap satelit (jumlah ke semua UT)
    """
    return predicted_power.sum(dim=1)  # [M, K] → [M]
