import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sentence_transformers import SentenceTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_FOLDS = 5
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

df = pd.read_csv("/content/train.csv")
llava_df = pd.read_csv("/content/train_llava_outputs_final_done.csv")

df["image_name"] = df["image_link"].apply(lambda x: x.split("/")[-1])
df = df.merge(llava_df, on="image_name", how="left")

def clean_text(x):
    if not isinstance(x, str):
        return ""
    return re.sub(r"\s+", " ", x).strip()

df["full_text"] = (
    df["catalog_content"].fillna("").apply(clean_text)
    + " "
    + df["llava_text"].fillna("").apply(clean_text)
)

df["label"] = np.log1p(df["price"])
y = df["label"].values

def extract_advanced_features(text):
    features = {
        "weight_grams": 0.0,
        "pack_size": 1,
        "total_weight": 0.0,
        "has_dimensions": 0,
        "number_count": 0
    }

    if not isinstance(text, str):
        return features

    text = text.lower()
    features["number_count"] = len(re.findall(r"\d+\.?\d*", text))

    m = re.search(r"(\d+\.?\d*)\s*(kg|g)", text)
    if m:
        w = float(m.group(1))
        features["weight_grams"] = w * (1000 if m.group(2) == "kg" else 1)

    m = re.search(r"pack\s*of\s*(\d+)", text)
    if m:
        features["pack_size"] = int(m.group(1))

    features["total_weight"] = features["weight_grams"] * features["pack_size"]
    features["has_dimensions"] = int(bool(re.search(r"\d+\s*x\s*\d+", text)))

    return features

num_df = df["full_text"].apply(extract_advanced_features).apply(pd.Series)

num_df["log_weight"] = np.log1p(num_df["weight_grams"])
num_df["log_total_weight"] = np.log1p(num_df["total_weight"])

num_features = num_df[
    ["log_weight", "log_total_weight",
     "pack_size", "has_dimensions", "number_count"]
].values

num_features = StandardScaler().fit_transform(num_features)

text_encoder = SentenceTransformer(
    "all-mpnet-base-v2", device=DEVICE
)

text_embeddings = text_encoder.encode(
    df["full_text"].tolist(),
    batch_size=128,
    show_progress_bar=True
)

class TextNumDataset(Dataset):
    def __init__(self, txt, num, y):
        self.txt = torch.tensor(txt, dtype=torch.float32)
        self.num = torch.tensor(num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.txt[i], self.num[i], self.y[i]

class TextNumericRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.txt_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, 512)
        )

        self.num_proj = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(512 + 128, 768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, txt, num):
        txt = self.txt_proj(txt)
        num = self.num_proj(num)
        x = torch.cat([txt, num], dim=1)
        return self.head(x).squeeze(1)

criterion = nn.SmoothL1Loss()  # Huber

def smape(y_true, y_pred):
    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred)
    return np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_pred) + np.abs(y_true) + 1e-8)
    ) * 100

from sklearn.model_selection import KFold
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(text_embeddings)):
    print(f"\n================ FOLD {fold+1}/{N_FOLDS} ================\n")

    X_txt_tr, X_txt_va = text_embeddings[tr_idx], text_embeddings[va_idx]
    X_num_tr, X_num_va = num_features[tr_idx], num_features[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    train_loader = DataLoader(
        TextNumDataset(X_txt_tr, X_num_tr, y_tr),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TextNumDataset(X_txt_va, X_num_va, y_va),
        batch_size=BATCH_SIZE
    )

    model = TextNumericRegressor().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_fold_smape = float("inf")

    for epoch in range(EPOCHS):
        # -------- TRAIN --------
        model.train()
        train_preds, train_labels = [], []

        for txt, num, yb in train_loader:
            txt, num, yb = txt.to(DEVICE), num.to(DEVICE), yb.to(DEVICE)

            out = model(txt, num)
            loss = criterion(out, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_preds.append(out.detach().cpu().numpy())
            train_labels.append(yb.cpu().numpy())

        # -------- VALIDATION --------
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for txt, num, yb in val_loader:
                txt, num, yb = txt.to(DEVICE), num.to(DEVICE), yb.to(DEVICE)
                out = model(txt, num)

                val_preds.append(out.cpu().numpy())
                val_labels.append(yb.cpu().numpy())

        val_smape = smape(
            np.concatenate(val_labels),
            np.concatenate(val_preds)
        )

        if val_smape < best_fold_smape:
            best_fold_smape = val_smape

        print(
            f"Fold {fold+1} | "
            f"Epoch {epoch+1} | "
            f"Val SMAPE {val_smape:.2f}"
        )

    fold_scores.append(best_fold_smape)
    print(f"🏆 Fold {fold+1} Best SMAPE: {best_fold_smape:.2f}")

print("\n================ FINAL RESULT ================")
print("Fold SMAPEs:", fold_scores)
print(f"Mean CV SMAPE: {np.mean(fold_scores):.2f}")
print(f"Std CV SMAPE:  {np.std(fold_scores):.2f}")
