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

    download_images_from_csv("train.csv", "train_images")

    download_images_from_csv("test.csv", "test_images")

    print("All downloads complete. You can now zip the folders for Kaggle upload.")

# INSTALL REQUIRED PACKAGES (run once)
"""
pip install transformers torch torchvision pandas numpy scikit-learn tqdm pillow
"""

import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import DistilBertTokenizer, DistilBertModel
from torchvision import models, transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def remove_price_leakage(text):
    """Remove all price mentions from text"""
    if pd.isna(text) or text == "":
        return ""

    text = str(text)

    # Remove price patterns
    text = re.sub(r'Price:\s*\$?[\d,]+\.?\d*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'List\s*Price:\s*\$?[\d,]+\.?\d*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Sale\s*Price:\s*\$?[\d,]+\.?\d*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'MSRP:\s*\$?[\d,]+\.?\d*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\$[\d,]+\.?\d+', '', text)
    text = re.sub(r'(?:USD|Rs\.?|INR|EUR|GBP)\s*[\d,]+\.?\d*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def load_ocr_from_csv(ocr_csv_path):
    """Load pre-extracted OCR results from CSV file"""
    if not os.path.exists(ocr_csv_path):
        print(f"⚠️ Warning: OCR file not found: {ocr_csv_path}")
        print("Continuing without OCR data...")
        return {}

    print(f"Loading OCR data from: {ocr_csv_path}")
    ocr_df = pd.read_csv(ocr_csv_path)

    if 'image_path' in ocr_df.columns and 'ocr_text' in ocr_df.columns:
        ocr_dict = dict(zip(ocr_df['image_path'], ocr_df['ocr_text']))
    elif 'image_link' in ocr_df.columns and 'extracted_text' in ocr_df.columns:
        ocr_dict = dict(zip(ocr_df['image_link'], ocr_df['extracted_text']))
    else:
        print(f"Available columns: {ocr_df.columns.tolist()}")
        ocr_dict = dict(zip(ocr_df.iloc[:, 0], ocr_df.iloc[:, 1]))

    ocr_dict = {k: remove_price_leakage(str(v)) for k, v in ocr_dict.items()}

    print(f"✓ Loaded {len(ocr_dict)} OCR entries")


def extract_numerical_features(text):
    """Extract numerical features like weight, dimensions, quantity"""
    features = {}

    if pd.isna(text):
        return {'has_weight': 0, 'has_dimensions': 0, 'has_quantity': 0, 'number_count': 0}

    text = str(text).lower()

    numbers = re.findall(r'\d+\.?\d*', text)
    features['number_count'] = len(numbers)

    weight_patterns = [r'\d+\.?\d*\s*(?:kg|g|lb|oz|pounds|grams)',
                      r'weight[:\s]+\d+']
    features['has_weight'] = int(any(re.search(p, text) for p in weight_patterns))

    dim_patterns = [r'\d+\.?\d*\s*x\s*\d+\.?\d*',
                   r'(?:height|width|length|size)[:\s]+\d+']
    features['has_dimensions'] = int(any(re.search(p, text) for p in dim_patterns))

    qty_patterns = [r'pack\s+of\s+\d+', r'\d+\s*(?:pack|count|piece|pcs)',
                   r'quantity[:\s]+\d+']
    features['has_quantity'] = int(any(re.search(p, text) for p in qty_patterns))

    return features


def extract_brand_features(text):
    """Extract brand-related features"""
    if pd.isna(text):
        return {'brand_length': 0, 'has_brand': 0}

    text = str(text)

    brand_keywords = ['by ', 'brand:', 'manufacturer:', 'made by']
    has_brand = int(any(keyword in text.lower() for keyword in brand_keywords))

    words = text.split()
    brand_length = len(words[0]) if words else 0

    return {'brand_length': brand_length, 'has_brand': has_brand}


class MultimodalDataset(Dataset):
    """Dataset combining text, OCR, and images"""

    def __init__(self, df, image_dir, tokenizer, img_transform, ocr_dict=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.img_transform = img_transform
        self.ocr_dict = ocr_dict or {}
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        catalog_text = str(row['product_text'])

        img_name = str(row['image_link']).split('/')[-1]
        img_path = os.path.join(self.image_dir, img_name)

        ocr_text = ""
        for possible_key in [img_path, img_name, str(row['image_link'])]:
            if possible_key in self.ocr_dict:
                ocr_text = self.ocr_dict[possible_key]
                break

        combined_text = f"{catalog_text} {ocr_text}".strip()

        tokens = self.tokenizer(
            combined_text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)

        num_features = extract_numerical_features(catalog_text)
        brand_features = extract_brand_features(catalog_text)

        text_features = {
            'text_length': len(catalog_text),
            'word_count': len(catalog_text.split()),
            'ocr_length': len(ocr_text),
            'ocr_word_count': len(ocr_text.split())
        }

        all_features = {**num_features, **brand_features, **text_features}
        feature_vector = torch.tensor(list(all_features.values()), dtype=torch.float32)

        try:
            image = Image.open(img_path).convert('RGB')
            image = self.img_transform(image)
        except:
            image = torch.zeros(3, 224, 224)

        if self.is_train:
            label = torch.tensor(row['label'], dtype=torch.float32)
            return input_ids, attention_mask, feature_vector, image, label
        else:
            return input_ids, attention_mask, feature_vector, image
        
class MultimodalPricePredictor(nn.Module):
    """
    Combines:
    - DistilBERT for text embeddings (catalog + OCR)
    - ResNet50 for image features
    - Hand-crafted numerical features
    - MLP fusion layer for final prediction
    """

    def __init__(self, num_features=10, freeze_text=True, freeze_vision=True):
        super().__init__()

        print("Loading DistilBERT...")
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')

        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

            for param in self.text_encoder.transformer.layer[-2:].parameters():
                param.requires_grad = True

        text_dim = 768  

        print("Loading ResNet50...")
        resnet = models.resnet50(pretrained=True)

        if freeze_vision:
            for param in resnet.parameters():
                param.requires_grad = False

        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])
        vision_dim = 2048  

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.vision_projection = nn.Sequential(
            nn.Linear(vision_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.feature_projection = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Total: 256 (text) + 256 (vision) + 64 (features) = 576
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256 + 64, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 1)
        )

        print("✓ Model initialized!")

    def forward(self, input_ids, attention_mask, features, images):
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embedding = text_outputs.last_hidden_state[:, 0, :] 
        text_feat = self.text_projection(text_embedding) 

        vision_embedding = self.vision_encoder(images) 
        vision_embedding = vision_embedding.view(vision_embedding.size(0), -1) 
        vision_feat = self.vision_projection(vision_embedding)  

        feat_proj = self.feature_projection(features)

        combined = torch.cat([text_feat, vision_feat, feat_proj], dim=1) 
        output = self.fusion(combined) 

        return output.squeeze(1)

def save_checkpoint(epoch, model, optimizer, scheduler, train_smape, val_smape,
                   best_val_smape, checkpoint_dir='checkpoints'):
    """Save comprehensive checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'train_smape': train_smape,
        'val_smape': val_smape,
        'best_val_smape': best_val_smape,
    }

    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    torch.save(checkpoint, latest_path)

    epoch_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, epoch_path)

    print(f"✓ Checkpoint saved: {latest_path}")

    return latest_path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load checkpoint and return epoch info"""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, float('inf')

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state'])

    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    if scheduler is not None and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])

    epoch = checkpoint.get('epoch', 0)
    best_val_smape = checkpoint.get('best_val_smape', float('inf'))

    print(f"✓ Resumed from epoch {epoch}")
    print(f"✓ Best val SMAPE so far: {best_val_smape:.2f}%")

    return epoch, best_val_smape

def smape(y_pred, y_true, eps=1e-8):
    """Symmetric Mean Absolute Percentage Error"""
    return torch.mean(
        2.0 * torch.abs(y_pred - y_true) /
        (torch.abs(y_pred) + torch.abs(y_true) + eps)
    ) * 100


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_smape = 0

    for input_ids, attention_mask, features, images, labels in tqdm(loader, desc="Training"):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        features = features.to(device)
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        preds = model(input_ids, attention_mask, features, images)
        loss = criterion(preds, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item()

        with torch.no_grad():
            preds_price = torch.expm1(preds)
            labels_price = torch.expm1(labels)
            total_smape += smape(preds_price, labels_price).item()

    return total_loss / len(loader), total_smape / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate on validation set"""
    model.eval()
    total_loss = 0
    total_smape = 0

    for input_ids, attention_mask, features, images, labels in tqdm(loader, desc="Validation"):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        features = features.to(device)
        images = images.to(device)
        labels = labels.to(device)

        preds = model(input_ids, attention_mask, features, images)
        loss = criterion(preds, labels)

        total_loss += loss.item()

        preds_price = torch.expm1(preds)
        labels_price = torch.expm1(labels)
        total_smape += smape(preds_price, labels_price).item()

    return total_loss / len(loader), total_smape / len(loader)

def main():

    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 2e-5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TRAIN_OCR_CSV = '/content/easyocr_train_cache (1) (1).csv'  
    TEST_OCR_CSV = ''

    CHECKPOINT_DIR = 'checkpoints'
    RESUME_FROM = '/content/checkpoint_epoch_6 (1).pt' 

    print(f"Using device: {DEVICE}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")

    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print("Cleaning price leakage from text...")
    train_df['product_text'] = train_df['catalog_content'].apply(remove_price_leakage)
    test_df['product_text'] = test_df['catalog_content'].apply(remove_price_leakage)

    train_df['label'] = np.log1p(train_df['price'])

    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)

    print(f"✓ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print("\n" + "="*60)
    print("LOADING PRE-EXTRACTED OCR FROM CSV")
    print("="*60)

    ocr_train_dict = load_ocr_from_csv(TRAIN_OCR_CSV)
    ocr_val_dict = ocr_train_dict 

    if os.path.exists(TEST_OCR_CSV):
        ocr_test_dict = load_ocr_from_csv(TEST_OCR_CSV)
    else:
        print(f"⚠️ Test OCR file not found: {TEST_OCR_CSV}")
        print("→ Test predictions will use catalog text only (no OCR)")
        ocr_test_dict = {}

    print("\n" + "="*60)
    print("PREPARING MODEL COMPONENTS")
    print("="*60)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MultimodalDataset(
        train_df, 'train_images', tokenizer, img_transform, ocr_train_dict, is_train=True
    )

    val_dataset = MultimodalDataset(
        val_df, 'train_images', tokenizer, img_transform, ocr_val_dict, is_train=True
    )

    test_dataset = MultimodalDataset(
        test_df, 'test_images', tokenizer, img_transform, ocr_test_dict, is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = MultimodalPricePredictor(num_features=10).to(DEVICE)

    criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW([
        {'params': model.text_encoder.parameters(), 'lr': LEARNING_RATE},
        {'params': model.vision_encoder.parameters(), 'lr': LEARNING_RATE / 10},
        {'params': model.text_projection.parameters(), 'lr': LEARNING_RATE * 5},
        {'params': model.vision_projection.parameters(), 'lr': LEARNING_RATE * 5},
        {'params': model.feature_projection.parameters(), 'lr': LEARNING_RATE * 5},
        {'params': model.fusion.parameters(), 'lr': LEARNING_RATE * 5}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    start_epoch = 0
    best_val_smape = float('inf')

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        start_epoch, best_val_smape = load_checkpoint(
            RESUME_FROM, model, optimizer, scheduler
        )
        start_epoch += 1  # Start from next epoch
    else:
        print("Starting training from scratch...")

    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    patience = 4
    patience_counter = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        train_loss, train_smape = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        val_loss, val_smape = validate(model, val_loader, criterion, DEVICE)

        print(f"\nTrain Loss: {train_loss:.4f} | Train SMAPE: {train_smape:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val SMAPE:   {val_smape:.2f}%")

        scheduler.step(val_smape)

        save_checkpoint(
            epoch, model, optimizer, scheduler,
            train_smape, val_smape, best_val_smape,
            checkpoint_dir=CHECKPOINT_DIR
        )

        if val_smape < best_val_smape:
            best_val_smape = val_smape
            patience_counter = 0

            best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_smape': val_smape,
            }, best_model_path)
            print(f"✓ NEW BEST MODEL! Val SMAPE: {val_smape:.2f}%")
            print(f"✓ Saved to: {best_model_path}")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
                break

    print("\n" + "="*60)
    print("GENERATING TEST PREDICTIONS")
    print("="*60)

    best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
    if os.path.exists(best_model_path):
        best_ckpt = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(best_ckpt['model_state'])
        print(f"✓ Loaded best model from epoch {best_ckpt['epoch']}")

    model.eval()
    predictions = []

    with torch.no_grad():
        for input_ids, attention_mask, features, images in tqdm(test_loader):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            features = features.to(DEVICE)
            images = images.to(DEVICE)

            preds = model(input_ids, attention_mask, features, images)
            preds_price = torch.expm1(preds)

            predictions.extend(preds_price.cpu().numpy())

    submission_df = pd.DataFrame({
        'index': test_df.index,
        'price': predictions
    })

    submission_df.to_csv('submission_final.csv', index=False)

    print(f"\n✓ Submission saved to submission_final.csv")
    print(f"✓ Best validation SMAPE: {best_val_smape:.2f}%")
    print(f"✓ Total predictions: {len(predictions)}")
    print(f"\nFirst 5 predictions:")
    print(submission_df.head())

    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()