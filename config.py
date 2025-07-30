# Nama file: config.py
# Pusat semua konfigurasi untuk pipeline dan dashboard.

# --- Konfigurasi Pipeline ---
SATELLITE_LIST = [1, 2, 6, 10]
NUM_SAMPLES = 1000
UT_COUNT = 5

# --- Konfigurasi Simulasi & Pengujian ---
R_MIN_THRESHOLD = 0.35  # <-- Nilai Rk_min Anda di sini
P_MAX = 300.0

# --- Direktori (Opsional, tapi rapi) ---
# Anda juga bisa memusatkan definisi path di sini
MODELS_DIR = "models"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
RESULTS_MODEL_DIR = "results/model_results"
RESULTS_BASELINE_DIR = "results/baseline_results"

print("✅ Konfigurasi berhasil dimuat.")