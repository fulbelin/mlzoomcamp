import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from src.model import get_model
from src.utils import ISICDataset, load_and_split, tfm
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)

train_df, val_df, small_dir = load_and_split()

train_data = ISICDataset(train_df, small_dir, tfm)
val_data   = ISICDataset(val_df, small_dir, tfm)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=8)
val_loader   = DataLoader(val_data, batch_size=128, shuffle=False,  num_workers=8)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

def evaluate():
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            preds.extend(model(x).squeeze().cpu().numpy())
            targets.extend(y.cpu().numpy())
    return roc_auc_score(targets, preds)

for epoch in range(3):
    model.train()
    for x,y in tqdm(train_loader):
        x,y = x.to(device),y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x).squeeze(),y)
        loss.backward()
        optimizer.step()

    print("Epoch:", epoch+1, "AUC:", evaluate())

torch.save(model.state_dict(), "models/isic_fast_model.pt")
print("Model saved!")
