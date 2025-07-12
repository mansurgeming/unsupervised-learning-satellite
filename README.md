# unsupervised-learning-satellite

Struktur repository

```
leo-power-opt/
├── data/
│   ├── raw/                   # Data fitur + target sebelum preprocessing
│   ├── processed/             # Dataset siap training (CSV/NPZ)
│   └── results/               # Evaluasi hasil model (.csv, .json, grafik)
│
├── models/
│   ├── checkpoints/           # File model terlatih (.pt / .h5)
│   ├── architecture.py        # PowerAllocatorModel
│   └── loss_fn.py             # Custom loss: QoS, power, handover, dll
│
├── src/
│   ├── build_dataset.py       # Generate data latih: fitur + target p_mk
│   ├── trainer.py             # Training loop + logging
│   ├── evaluate.py            # Evaluasi model: rate, handover, power
│   ├── simulate.py            # Run trained model di skenario baru
│   ├── channel_model.py       # Hitung fitur channel (bisa dummy / skyfield)
│   └── utils/
│       ├── geo.py             # Generate UT positions dari center + radius
│       └── config.py          # Konstanta global (area, frekuensi, target QoS, dll)
│
├── notebooks/
│   └── analysis.ipynb         # EDA, grafik rate, distribusi daya, handover
│
├── tests/
│   └── test_loss_fn.py        # Unit test loss, model output, dll
│
├── requirements.txt
├── README.md
└── run.sh                     # Script batch: build → train → eval (opsional)
```
