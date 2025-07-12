import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# === Config ===
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3

# === Dataset ===
class PowerDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        grouped = df.groupby("sample_id")
        self.X = []
        for _, group in grouped:
            features = group[[
                "dist_km", "elev_deg", "azim_deg",
                "fspl_db", "shadowing_db"
            ]].values  # shape (10, 5)
            self.X.append(features)
        self.X = torch.tensor(self.X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]  # (10, 5)

# === Model ===
class PowerAllocatorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.ReLU()  # output: power per UT (>= 0)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten (10, 5) → (50,)
        return self.mlp(x)  # output: (batch_size, 10)

# === Dummy Loss (placeholder for QoS-based loss) ===
def dummy_unsupervised_loss(pred):
    # just penalize total power (as placeholder)
    return torch.mean(torch.sum(pred, dim=1))

# === Training Loop ===
def train():
    train_ds = PowerDataset("data/processed/train_data.csv")
    val_ds = PowerDataset("data/processed/val_data.csv")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = PowerAllocatorModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x in train_loader:
            pred = model(x)
            loss = dummy_unsupervised_loss(pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for x in val_loader:
                pred = model(x)
                loss = dummy_unsupervised_loss(pred)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {total_loss:.2f} | Val Loss: {val_loss:.2f}")

    torch.save(model.state_dict(), "model/power_allocator.pt")
    print("[DONE] Model saved to model/power_allocator.pt")

if __name__ == "__main__":
    train()