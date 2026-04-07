import os
import pandas as pd
import urllib.request
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import time

MAX_RETRIES = 3
TIMEOUT = 10

def download_image(image_link, savefolder, max_retries=MAX_RETRIES):
    if not isinstance(image_link, str):
        return None

    filename = Path(image_link).name
    image_save_path = os.path.join(savefolder, filename)

    if os.path.exists(image_save_path):
        return None  

    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(image_link, image_save_path)
            return None 
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1) 
            else:
                return image_link 

def download_images_from_csv(csv_path, savefolder, failed_csv="failed_images.csv"):
    df = pd.read_csv(csv_path)

    if 'image_link' not in df.columns:
        raise ValueError(f"'image_link' column not found in {csv_path}")

    image_links = df['image_link'].dropna().tolist()

    os.makedirs(savefolder, exist_ok=True)

    partial_download = partial(download_image, savefolder=savefolder)

    failed_links = []

    with Pool(100) as pool:
        for result in tqdm(
            pool.imap_unordered(partial_download, image_links),
            total=len(image_links)
        ):
            if result is not None:
                failed_links.append(result)

    if failed_links:
        pd.DataFrame({'image_link': failed_links}).to_csv(failed_csv, index=False)
        print(f"⚠️ {len(failed_links)} images failed. Saved to {failed_csv}")
    else:
        print("✅ All images downloaded successfully!")

def retry_failed_images(failed_csv, savefolder):
    if not os.path.exists(failed_csv):
        print("No failed CSV found. Nothing to retry.")
        return

    df = pd.read_csv(failed_csv)
    links = df['image_link'].dropna().tolist()

    if not links:
        print("No failed images to retry.")
        return

    print(f"Retrying {len(links)} failed images...")

    partial_download = partial(download_image, savefolder=savefolder)

    still_failed = []

    with Pool(50) as pool:
        for result in tqdm(
            pool.imap_unordered(partial_download, links),
            total=len(links)
        ):
            if result is not None:
                still_failed.append(result)

    if still_failed:
        pd.DataFrame({'image_link': still_failed}).to_csv(failed_csv, index=False)
        print(f"Still failed: {len(still_failed)} images")
    else:
        os.remove(failed_csv)
        print("All previously failed images downloaded!")

if __name__ == "__main__":

    download_images_from_csv("train.csv", "train_images")
    retry_failed_images("failed_images.csv", "train_images")
    print("Download process complete. Ready for Kaggle upload.")

import pandas as pd
import numpy as np
import re

df = pd.read_csv("train.csv")

df["image_name"] = df["image_link"].apply(
    lambda x: x.split("/")[-1] if isinstance(x, str) else ""
)
import os
IMAGE_FOLDER = "/content/train_images"
existing_images = set(os.listdir(IMAGE_FOLDER))
df = df[df["image_name"].isin(existing_images)].reset_index(drop=True)

import os
import re
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import faiss

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_FOLDER  = "/content/train_images" 
EMBED_DIM     = 512
BATCH_SIZE    = 32
EPOCHS        = 30
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
TEMPERATURE   = 10.0
KNN_K         = 30
KNN_ALPHA     = 30.0
N_FOLDS       = 5
VAL_SIZE      = 0.1
RANDOM_STATE  = 42

BRAND_LIST = [
    'nescafe','starbucks','keurig','dunkin','lavazza',"peet's",
    'folgers','tassimo','samsung','apple','sony','lg','panasonic',
    'bose','dell','hp','lenovo','acer','microsoft','nike','adidas',
    'under armour','puma','reebok','new balance','champion','lego',
    'hasbro','mattel','nerf','funko','play-doh','amazonbasics',
    'kirkland signature','great value','up&up','logitech'
]

def parse_content(content_string):
    if not isinstance(content_string, str):
        content_string = ""
    lines = content_string.strip().split('\n')
    value, unit, brand = 1.0, "Unknown", "Unknown"
    item_name, bullet_points = "", []

    for line in lines:
        lower = line.lower()
        if lower.startswith("item name:"):
            item_name = line[len("item name:"):].strip()
        elif lower.startswith("bullet point"):
            bullet_points.append(
                re.sub(r'Bullet Point \d+:', '', line, flags=re.IGNORECASE).strip()
            )
        elif lower.startswith("value:"):
            try:
                value = float(line[len("value:"):].strip())
            except:
                value = 1.0
        elif lower.startswith("unit:"):
            unit = line[len("unit:"):].strip()

    text_for_brand = (
        item_name + " " + (bullet_points[0] if bullet_points else "")
    ).lower()
    for b in BRAND_LIST:
        if f" {b} " in f" {text_for_brand} ":
            brand = b
            break

    clean_text = " ".join([item_name] + bullet_points)
    return pd.Series(
        [value, unit, brand, clean_text],
        index=["quantity", "unit", "brand", "clean_text"]
    )


def prepare_dataframe(csv_path, image_folder):
    df = pd.read_csv(csv_path)
    df["image_name"] = df["image_link"].apply(
        lambda x: x.split("/")[-1] if isinstance(x, str) else ""
    )
    existing = set(os.listdir(image_folder))
    df = df[df["image_name"].isin(existing)].reset_index(drop=True)
    df = pd.concat([df, df["catalog_content"].apply(parse_content)], axis=1)
    print(f"  Loaded {len(df)} rows from {csv_path}")
    return df

def split_and_build_features(df):
    """
    Recreates original train_test_split with same random_state=42.
    Embeddings were computed on train_split rows → they align directly.
    """
    train_split, val_split = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
    )
    train_split = train_split.reset_index(drop=True)
    val_split   = val_split.reset_index(drop=True)

    scaler = StandardScaler()
    train_split["quantity_scaled"] = scaler.fit_transform(train_split[["quantity"]])
    val_split["quantity_scaled"]   = scaler.transform(val_split[["quantity"]])

    train_cats = pd.get_dummies(train_split[['unit', 'brand']], prefix=['unit', 'brand'])
    val_cats   = pd.get_dummies(val_split[['unit',  'brand']], prefix=['unit', 'brand'])
    train_cats, val_cats = train_cats.align(val_cats, join="outer", axis=1, fill_value=0)

    train_split = pd.concat([train_split, train_cats], axis=1)
    val_split   = pd.concat([val_split,   val_cats],   axis=1)
    manual_cols = ['quantity_scaled'] + list(train_cats.columns)

    print(f"  Train split : {len(train_split)} rows")
    print(f"  Val split   : {len(val_split)} rows")
    print(f"  Manual features: {len(manual_cols)}")

    return train_split, val_split, manual_cols, scaler

def make_group_ids(df):
    def _hash(row):
        s = str(row["catalog_content"]) + "|" + str(row["image_name"])
        return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)
    return df.apply(_hash, axis=1).values

class EmbeddingDataset(Dataset):
    def __init__(self, text_emb, image_emb, df, manual_cols):
        self.text      = text_emb
        self.image     = image_emb
        manual_np      = df[manual_cols].fillna(0).astype(np.float32).values
        self.manual    = torch.from_numpy(manual_np)
        self.log_price = torch.log1p(
            torch.tensor(df["price"].values, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.log_price)

    def __getitem__(self, idx):
        return (
            self.text[idx],
            self.image[idx],
            self.manual[idx],
            self.log_price[idx],
        )

class StrongFusionEmbedder(nn.Module):
    def __init__(self, text_dim, image_dim, manual_dim, embed_dim=512):
        super().__init__()
        hidden = 1024

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=16, dropout=0.2, batch_first=True
        )
        self.manual_proj = nn.Sequential(
            nn.Linear(manual_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden + 256, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim),
        )

    def forward(self, text_emb, image_emb, manual, normalize=True):
        text  = self.text_proj(text_emb.float())
        image = self.image_proj(image_emb.float())
        attn_out, _ = self.attn(
            text.unsqueeze(1),
            image.unsqueeze(1),
            image.unsqueeze(1),
        )
        fused  = attn_out.squeeze(1) + text
        manual = self.manual_proj(manual.float())
        out    = self.mlp(torch.cat([fused, manual], dim=1))
        if normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out

class PriceProximityLoss(nn.Module):
    def __init__(self, temperature=10.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, log_prices):
        sim    = torch.mm(embeddings, embeddings.t())
        pdiff  = torch.abs(log_prices.unsqueeze(1) - log_prices.unsqueeze(0))
        target = torch.exp(-pdiff / self.temperature)
        mask   = ~torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
        return F.mse_loss(sim[mask], target[mask])


def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float32), eps)
    num    = np.abs(y_pred - y_true)
    den    = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(num / (den + eps)) * 100)

def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index


def knn_predict(query_embs, index, train_log_prices, k=30, alpha=30.0):
    sims, ids = index.search(query_embs.astype(np.float32), k)
    x  = sims * alpha
    x -= x.max(axis=-1, keepdims=True)
    w  = np.exp(x)
    w /= w.sum(axis=-1, keepdims=True)
    return np.expm1((w * train_log_prices[ids]).sum(axis=1)).astype(np.float32)

@torch.no_grad()
def extract_embeddings(model, text_emb, image_emb, df, manual_cols, batch_size=256):
    model.eval()
    manual_np = df[manual_cols].fillna(0).astype(np.float32).values
    all_embs  = []
    for i in range(0, len(df), batch_size):
        t  = text_emb[i:i+batch_size].to(DEVICE)
        im = image_emb[i:i+batch_size].to(DEVICE)
        m  = torch.from_numpy(manual_np[i:i+batch_size]).to(DEVICE)
        all_embs.append(model(t, im, m, normalize=True).cpu().numpy())
    return np.vstack(all_embs).astype(np.float32)

def train_fold(fold, tr_idx, fv_idx,
               text_emb, image_emb, train_split, manual_cols):
    print(f"\n{'='*60}")
    print(f"FOLD {fold}  |  train={len(tr_idx)}  fold-val={len(fv_idx)}")
    print(f"{'='*60}")

    tr_df = train_split.iloc[tr_idx].reset_index(drop=True)
    fv_df = train_split.iloc[fv_idx].reset_index(drop=True)

    tr_loader = DataLoader(
        EmbeddingDataset(text_emb[tr_idx], image_emb[tr_idx], tr_df, manual_cols),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
    )

    model = StrongFusionEmbedder(
        text_dim   = text_emb.shape[1],
        image_dim  = image_emb.shape[1],
        manual_dim = len(manual_cols),
        embed_dim  = EMBED_DIM,
    ).to(DEVICE)

    criterion = PriceProximityLoss(temperature=TEMPERATURE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_smape_val = float("inf")
    best_ckpt      = None
    tr_log_prices  = np.log1p(tr_df["price"].values.astype(np.float32))

    for epoch in range(1, EPOCHS + 1):

        model.train()
        total_loss = 0.0
        for txt, img, manual, log_price in tr_loader:
            txt, img, manual, log_price = (
                txt.to(DEVICE), img.to(DEVICE),
                manual.to(DEVICE), log_price.to(DEVICE),
            )
            optimizer.zero_grad()
            emb  = model(txt, img, manual, normalize=True)
            loss = criterion(emb, log_price)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        tr_embs = extract_embeddings(model, text_emb[tr_idx], image_emb[tr_idx], tr_df, manual_cols)
        fv_embs = extract_embeddings(model, text_emb[fv_idx], image_emb[fv_idx], fv_df, manual_cols)
        index   = build_faiss_index(tr_embs)
        preds   = knn_predict(fv_embs, index, tr_log_prices, k=KNN_K, alpha=KNN_ALPHA)
        ep_smape = smape(fv_df["price"].values, preds)

        print(
            f"  Epoch {epoch:3d} | "
            f"loss={total_loss/len(tr_loader):.6f} | "
            f"fold-val SMAPE={ep_smape:.4f}%"
        )

        if ep_smape < best_smape_val:
            best_smape_val = ep_smape
            best_ckpt = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    ✓ Best saved ({best_smape_val:.4f}%)")

    model.load_state_dict(best_ckpt)
    print(f"\nFold {fold} best SMAPE: {best_smape_val:.4f}%")
    return model, best_smape_val

def ensemble_predict(fold_models, train_text, train_image, train_df,
                     query_text, query_image, query_df, manual_cols):
    train_log_prices = np.log1p(train_df["price"].values.astype(np.float32))
    log_preds_list   = []
    for fold, model in enumerate(fold_models):
        print(f"  Fold {fold} predicting...")
        tr_embs    = extract_embeddings(model, train_text,  train_image,  train_df,  manual_cols)
        query_embs = extract_embeddings(model, query_text,  query_image,  query_df,  manual_cols)
        index      = build_faiss_index(tr_embs)
        preds      = knn_predict(query_embs, index, train_log_prices, k=KNN_K, alpha=KNN_ALPHA)
        log_preds_list.append(np.log1p(preds))
    ensemble_log = np.stack(log_preds_list, axis=0).mean(axis=0)
    return np.expm1(ensemble_log).astype(np.float32)

if __name__ == "__main__":

    print("Loading data...")
    df = prepare_dataframe("train.csv", IMAGE_FOLDER)

    print("\nSplitting and building manual features...")
    train_split, val_split, manual_cols, scaler = split_and_build_features(df)
    print("\nLoading pre-computed embeddings...")

    train_text_data  = torch.load(
        "train_text_embeddings_SFR-Embedding-Mistral(7B).pt", weights_only=False
    )
    train_image_data = torch.load(
        "train_image_embeddings_EVA02-E-14-plus.pt", weights_only=False
    )
    val_text_data    = torch.load(
        "val_text_embeddings_SFR-Embedding-Mistral(7B).pt", weights_only=False
    )
    val_image_data   = torch.load(
        "val_image_embeddings_EVA02-E-14-plus.pt", weights_only=False
    )

    train_text_emb  = train_text_data["embeddings"]    # (67499, text_dim)
    train_image_emb = train_image_data["embeddings"]   # (67499, image_dim)
    val_text_emb    = val_text_data["embeddings"]       # (7500,  text_dim)
    val_image_emb   = val_image_data["embeddings"]      # (7500,  image_dim)

    assert len(train_text_emb)  == len(train_split), \
        f"Train text emb {len(train_text_emb)} != train_split {len(train_split)}"
    assert len(train_image_emb) == len(train_split), \
        f"Train image emb {len(train_image_emb)} != train_split {len(train_split)}"
    assert len(val_text_emb)    == len(val_split), \
        f"Val text emb {len(val_text_emb)} != val_split {len(val_split)}"
    assert len(val_image_emb)   == len(val_split), \
        f"Val image emb {len(val_image_emb)} != val_split {len(val_split)}"

    print(f"  Train: text={train_text_emb.shape}  image={train_image_emb.shape}")
    print(f"  Val  : text={val_text_emb.shape}    image={val_image_emb.shape}")
    print("  ✓ All embeddings aligned")

    print("\nStarting GroupKFold training...")
    group_ids   = make_group_ids(train_split)
    gkf         = GroupKFold(n_splits=N_FOLDS)
    fold_models = []
    fold_smapes = []
    oof_preds   = np.zeros(len(train_split), dtype=np.float32)

    for fold, (tr_idx, fv_idx) in enumerate(
        gkf.split(np.zeros(len(train_split)), groups=group_ids)
    ):
        model, best_smape = train_fold(
            fold, tr_idx, fv_idx,
            train_text_emb, train_image_emb,
            train_split, manual_cols,
        )
        fold_models.append(model)
        fold_smapes.append(best_smape)

        tr_df_f = train_split.iloc[tr_idx].reset_index(drop=True)
        fv_df_f = train_split.iloc[fv_idx].reset_index(drop=True)
        tr_embs = extract_embeddings(model, train_text_emb[tr_idx], train_image_emb[tr_idx], tr_df_f, manual_cols)
        fv_embs = extract_embeddings(model, train_text_emb[fv_idx], train_image_emb[fv_idx], fv_df_f, manual_cols)
        tr_log  = np.log1p(tr_df_f["price"].values.astype(np.float32))
        oof_preds[fv_idx] = knn_predict(
            fv_embs, build_faiss_index(tr_embs), tr_log, k=KNN_K, alpha=KNN_ALPHA
        )
        torch.save(model.state_dict(), f"fusion_embedder_fold{fold}.pt")

    oof_smape = smape(train_split["price"].values, oof_preds)

    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for i, s in enumerate(fold_smapes):
        print(f"  Fold {i}: {s:.4f}%")
    print(f"  Mean : {np.mean(fold_smapes):.4f}% ± {np.std(fold_smapes):.4f}%")
    print(f"  OOF  : {oof_smape:.4f}%")

    print(f"\n{'='*60}")
    print("HELD-OUT EVALUATION  (val_split — SMAPE metric)")
    print(f"{'='*60}")

    val_preds = ensemble_predict(
        fold_models,
        train_text_emb, train_image_emb, train_split,
        val_text_emb,   val_image_emb,   val_split,
        manual_cols,
    )
    val_smape = smape(val_split["price"].values, val_preds)

    pd.DataFrame({
        "sample_id" : val_split["sample_id"],
        "price_true": val_split["price"],
        "price_pred": np.maximum(val_preds, 0.01),
    }).to_csv("val_predictions.csv", index=False)

    print(f"\n{'='*60}")
    print(f"  OOF SMAPE (train_split, 5-fold) : {oof_smape:.4f}%")
    print(f"  Val SMAPE (held-out 10%)        : {val_smape:.4f}%")
    print(f"  Gap                             : {val_smape - oof_smape:+.4f}%")
    print(f"{'='*60}")
    print(f"\nHeld-out Val SMAPE: {val_smape:.4f}%")