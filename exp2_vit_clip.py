import os
import pandas as pd
import urllib.request
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

def download_image(image_link, savefolder):
    if isinstance(image_link, str):
        filename = Path(image_link).name
        image_save_path = os.path.join(savefolder, filename)
        if not os.path.exists(image_save_path):
            try:
                urllib.request.urlretrieve(image_link, image_save_path)
            except Exception as ex:
                print(f"Warning: Not able to download - {image_link}\n{ex}")
    return

def download_images_from_csv(csv_path, savefolder):
    df = pd.read_csv(csv_path)
    if 'image_link' not in df.columns:
        raise ValueError(f"'image_link' column not found in {csv_path}")
    image_links = df['image_link'].dropna().tolist()

    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    partial_download = partial(download_image, savefolder=savefolder)
    with Pool(100) as pool:
        list(tqdm(pool.imap_unordered(partial_download, image_links), total=len(image_links)))

if __name__ == "__main__":
    # Download training images
    download_images_from_csv("train.csv", "train_images")

    # Download test images
    download_images_from_csv("test.csv", "test_images")

    # Download sample test images
    # download_images_from_csv("sample_test.csv", "sample_test_images")

    print("All downloads complete. You can now zip the folders for Kaggle upload.")

# pip install open_clip_torch pillow tqdm

import pandas as pd
import numpy as np

train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

train_df["product_text"] = train_df["catalog_content"]
train_df["label"] = np.log1p(train_df["price"])

test_df["product_text"] = test_df["catalog_content"]


import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class AmazonProductDataset(Dataset):
    def __init__(self, df, image_dir, tokenizer, image_preprocess, is_train=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_preprocess = image_preprocess
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = str(row["product_text"])
        text_tokens = self.tokenizer([text])[0]

        img_name = str(row["image_link"]).split("/")[-1]
        img_path = os.path.join(self.image_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.image_preprocess(image)
        except:
            image = torch.zeros(3, 224, 224)  

        if self.is_train:
            label = torch.tensor(row["label"], dtype=torch.float32)
            return text_tokens, image, label
        else:
            return text_tokens, image


import torch.nn as nn
import open_clip

class CLIPMultimodalRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-B-32",
            pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        # Freeze CLIP
        for p in self.clip.parameters():
            p.requires_grad = False

        embed_dim = self.clip.text_projection.shape[1]

        self.regressor = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, text, image):
        text_feat = self.clip.encode_text(text)
        image_feat = self.clip.encode_image(image)

        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        fused = torch.cat([text_feat, image_feat], dim=1)
        return self.regressor(fused).squeeze(1)

import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIPMultimodalRegressor().to(device)

train_dataset = AmazonProductDataset(
    train_df,
    image_dir="train_images",
    tokenizer=model.tokenizer,
    image_preprocess=model.preprocess,
    is_train=True
)

test_dataset = AmazonProductDataset(
    test_df,
    image_dir="test_images",
    tokenizer=model.tokenizer,
    image_preprocess=model.preprocess,
    is_train=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,        
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

import torch

def smape(y_pred, y_true, eps=1e-8):
    return torch.mean(
        2.0 * torch.abs(y_pred - y_true) /
        (torch.abs(y_pred) + torch.abs(y_true) + eps)
    ) * 100

import torch.optim as optim
from tqdm import tqdm

criterion = nn.SmoothL1Loss()

optimizer = optim.Adam(model.regressor.parameters(), lr=1e-3)

start_epoch = 0
num_epochs = 8

checkpoint_path = "/content/checkpoint_epoch_3.pt"  # change if needed

if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}")


for epoch in range(start_epoch, num_epochs):   # 🔧 small change
    model.train()
    running_loss = 0
    running_smape = 0

    for text, image, y in tqdm(train_loader):
        text = text.to(device)
        image = image.to(device)
        y = y.to(device)

        preds = model(text, image)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds_price = torch.expm1(preds)
        y_price = torch.expm1(y)
        running_smape += smape(preds_price, y_price).item()

    print(
        f"Epoch {epoch} | "
        f"Loss: {running_loss / len(train_loader):.4f} | "
        f"SMAPE: {running_smape / len(train_loader):.2f}"
    )

    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, f"checkpoint_epoch_{epoch}.pt")

