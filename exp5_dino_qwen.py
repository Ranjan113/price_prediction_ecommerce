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

    print(f"🔁 Retrying {len(links)} failed images...")

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
        print(f"⚠️ Still failed: {len(still_failed)} images")
    else:
        os.remove(failed_csv)
        print("✅ All previously failed images downloaded!")

if __name__ == "__main__":

    download_images_from_csv("train.csv", "train_images")
    retry_failed_images("failed_images.csv", "train_images")
    print("🎉 Download process complete. Ready for Kaggle upload.")

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

BRAND_LIST = [
    'nescafe','starbucks','keurig','dunkin','lavazza',"peet's",
    'folgers','tassimo','samsung','apple','sony','lg','panasonic',
    'bose','dell','hp','lenovo','acer','microsoft','nike','adidas',
    'under armour','puma','reebok','new balance','champion','lego',
    'hasbro','mattel','nerf','funko','play-doh','amazonbasics',
    'kirkland signature','great value','up&up','logitech'
]

def parse_content_with_brand(content_string):

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

    text_for_brand = (item_name + " " + (bullet_points[0] if bullet_points else "")).lower()

    for b in BRAND_LIST:
        if f" {b} " in f" {text_for_brand} ":
            brand = b
            break

    clean_text = " ".join([item_name] + bullet_points)

    return pd.Series(
        [value, unit, brand, clean_text],
        index=["quantity", "unit", "brand", "clean_text"]
    )

df = pd.concat([df, df["catalog_content"].apply(parse_content_with_brand)], axis=1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42
)

scaler = StandardScaler()

train_df["quantity_scaled"] = scaler.fit_transform(train_df[["quantity"]])
val_df["quantity_scaled"] = scaler.transform(val_df[["quantity"]])

train_cats = pd.get_dummies(train_df[['unit','brand']], prefix=['unit','brand'])
val_cats   = pd.get_dummies(val_df[['unit','brand']], prefix=['unit','brand'])

train_cats, val_cats = train_cats.align(val_cats, join="outer", axis=1, fill_value=0)

train_df = pd.concat([train_df, train_cats], axis=1)
val_df   = pd.concat([val_df, val_cats], axis=1)

manual_feature_columns = ['quantity_scaled'] + list(train_cats.columns)

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor
)

import torch
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Qwen/Qwen3-Embedding-8B"

print("Loading Qwen3-Embedding-8B...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

text_model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  
    device_map="auto",
    trust_remote_code=True
)

text_model.eval()

for p in text_model.parameters():
    p.requires_grad = False

print("Qwen3-Embedding-8B loaded successfully.")

from tqdm.auto import tqdm

def precompute_text_embeddings(df, batch_size=2):

    all_embeddings = []
    text_model.eval()

    total_batches = (len(df) + batch_size - 1) // batch_size

    with tqdm(total=total_batches, desc="Text Embeddings", leave=True) as pbar:

        for i in range(0, len(df), batch_size):

            batch = df.iloc[i:i+batch_size]

            texts = [
                row["clean_text"]
                for _, row in batch.iterrows()
            ]

            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad(), torch.autocast("cuda"):
                outputs = text_model(**inputs)

            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()

            summed = torch.sum(hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)

            embeddings = summed / counts
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)

            all_embeddings.append(embeddings.cpu())

            pbar.update(1)

    return torch.cat(all_embeddings).half()

train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)

train_text_embeddings = precompute_text_embeddings(train_df)
val_text_embeddings   = precompute_text_embeddings(val_df)

torch.save({
    "embeddings": train_text_embeddings,
    "price": torch.tensor(train_df["price"].values),
    "index": train_df.index.values
}, "train_text_embeddings_qwen8b.pt")

torch.save({
    "embeddings": val_text_embeddings,
    "price": torch.tensor(val_df["price"].values),
    "index": val_df.index.values
}, "val_text_embeddings_qwen8b.pt")

print("✅ Text embeddings saved successfully.")

import torch
from transformers import AutoImageProcessor, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "facebook/dinov2-giant"

print("Loading DINOv2-Giant...")

image_processor = AutoImageProcessor.from_pretrained(model_name)

image_model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).to(DEVICE)

image_model.eval()

for p in image_model.parameters():
    p.requires_grad = False

print("DINOv2-Giant loaded successfully.")
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from PIL import Image
import os

def precompute_image_embeddings(df, image_folder, batch_size=4):

    all_embeddings = []
    image_model.eval()

    total_batches = (len(df) + batch_size - 1) // batch_size

    with tqdm(total=total_batches,
              desc="Image Embeddings",
              dynamic_ncols=True,
              leave=True) as pbar:

        for i in range(0, len(df), batch_size):

            batch = df.iloc[i:i+batch_size]
            images = []

            for _, row in batch.iterrows():
                img_path = os.path.join(image_folder, row["image_name"])
                image = Image.open(img_path).convert("RGB")
                images.append(image)

            inputs = image_processor(
                images=images,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad(), torch.autocast("cuda"):
                outputs = image_model(**inputs)

            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = F.normalize(embeddings, dim=1)

            all_embeddings.append(embeddings.cpu())

            pbar.update(1)

    return torch.cat(all_embeddings).half()

train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)

train_image_embeddings = precompute_image_embeddings(train_df, IMAGE_FOLDER)
val_image_embeddings   = precompute_image_embeddings(val_df, IMAGE_FOLDER)

torch.save({
    "embeddings": train_image_embeddings,
    "price": torch.tensor(train_df["price"].values),
    "index": train_df.index.values
}, "train_image_embeddings_facebook/dinov2-giant.pt")

torch.save({
    "embeddings": val_image_embeddings,
    "price": torch.tensor(val_df["price"].values),
    "index": val_df.index.values
}, "val_image_embeddings_facebook/dinov2-giant.pt")

print("Image embeddings saved successfully.")

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class EmbeddingDataset(Dataset):

    def __init__(self,
                 text_embeddings,
                 image_embeddings,
                 prices,
                 df):

        self.text = text_embeddings
        self.image = image_embeddings

        manual_np = (
            df[manual_feature_columns]
            .fillna(0)
            .astype(np.float32)
            .values
        )

        self.manual = torch.from_numpy(manual_np)

        self.price = torch.log(
            torch.tensor(prices, dtype=torch.float32) + 1
        )

    def __len__(self):
        return len(self.price)

    def __getitem__(self, idx):
        return (
            self.image[idx],
            self.text[idx],
            self.manual[idx],
            self.price[idx]
        )

from torch.utils.data import DataLoader

train_dataset = EmbeddingDataset(
    train_text_embeddings,
    train_image_embeddings,
    train_df["price"].values,
    train_df
)

val_dataset = EmbeddingDataset(
    val_text_embeddings,
    val_image_embeddings,
    val_df["price"].values,
    val_df
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    pin_memory=True
)

import torch.nn as nn

class SMAPELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred_log, target_log):
        pred = torch.exp(pred_log) - 1
        target = torch.exp(target_log) - 1

        numerator = torch.abs(pred - target)
        denominator = (torch.abs(pred) + torch.abs(target)) / 2

        return torch.mean(numerator / (denominator + self.eps))

import torch.nn.functional as F

class StrongFusionModel(nn.Module):

    def __init__(self, text_dim, image_dim, manual_dim):
        super().__init__()

        hidden = 768

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=12,
            dropout=0.2,
            batch_first=True
        )

        self.manual_proj = nn.Sequential(
            nn.Linear(manual_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden + 512, 1024),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.GELU(),

            nn.Linear(256, 1)
        )

    def forward(self, text_emb, image_emb, manual):

        text = self.text_proj(text_emb.float())
        image = self.image_proj(image_emb.float())

        attn_out, _ = self.attn(
            text.unsqueeze(1),
            image.unsqueeze(1),
            image.unsqueeze(1)
        )

        fused = attn_out.squeeze(1) + text

        manual_feat = self.manual_proj(manual.float())

        combined = torch.cat([fused, manual_feat], dim=1)

        return self.mlp(combined).squeeze(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = StrongFusionModel(
    text_dim=train_text_embeddings.shape[1],
    image_dim=train_image_embeddings.shape[1],
    manual_dim=len(manual_feature_columns)
).to(DEVICE)

import torch.optim as optim

criterion = SMAPELoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2
)

best_smape = float("inf")

for epoch in range(200):

    model.train()
    total_loss = 0

    for img_emb, txt_emb, manual, price in train_loader:

        img_emb = img_emb.to(DEVICE)
        txt_emb = txt_emb.to(DEVICE)
        manual  = manual.to(DEVICE)
        price   = price.to(DEVICE)

        optimizer.zero_grad()

        preds = model(txt_emb, img_emb, manual)

        loss = criterion(preds, price)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    model.eval()
    val_smape = 0

    with torch.no_grad():
        for img_emb, txt_emb, manual, price in val_loader:

            img_emb = img_emb.to(DEVICE)
            txt_emb = txt_emb.to(DEVICE)
            manual  = manual.to(DEVICE)
            price   = price.to(DEVICE)

            preds = model(txt_emb, img_emb, manual)
            val_smape += criterion(preds, price).item()

    val_smape /= len(val_loader)

    print(f"Epoch {epoch+1} | Val SMAPE: {val_smape:.4f}")

    if val_smape < best_smape:
        best_smape = val_smape
        torch.save(model.state_dict(), "best_model.pt")
        print("✓ New Best Model Saved")

print("\nBest Validation SMAPE:", best_smape)
