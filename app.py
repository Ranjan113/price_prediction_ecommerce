import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
import os

device = "cpu"

# =========================
# MODEL DEFINITION
# =========================

class StrongFusionModel(nn.Module):

    def __init__(self, text_dim, image_dim, manual_dim):
        super().__init__()

        hidden = 1024

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
            num_heads=16,
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
            nn.Linear(hidden + 512, 2048),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, text_emb, image_emb, manual):

        text = self.text_proj(text_emb)
        image = self.image_proj(image_emb)

        attn_out, _ = self.attn(
            text.unsqueeze(1),
            image.unsqueeze(1),
            image.unsqueeze(1)
        )

        fused = attn_out.squeeze(1) + text

        manual_feat = self.manual_proj(manual)

        combined = torch.cat([fused, manual_feat], dim=1)

        return self.mlp(combined).squeeze(1)


# =========================
# LOAD DATA + MODEL
# =========================

@st.cache_resource
def load_all():

    text_data = torch.load("demo_text.pt", weights_only=False)
    image_data = torch.load("demo_image_30.pt", weights_only=False)

    # safety check
    assert len(text_data["text_emb"]) == len(image_data["image_emb"]), "Mismatch!"

    model = StrongFusionModel(
        text_dim=4096,
        image_dim=1024,
        manual_dim=117
    )

    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()

    return text_data, image_data, model


text_data, image_data, model = load_all()

# =========================
# PREDICTION FUNCTION
# =========================

def predict(text_emb, image_emb):

    manual = torch.zeros(1, 117)

    with torch.no_grad():
        pred_log = model(text_emb, image_emb, manual)

    pred_price = torch.exp(pred_log) - 1
    return float(pred_price.item())


# =========================
# MAIN UI
# =========================

st.set_page_config(page_title="Price Prediction Demo", layout="wide")

st.title("🛒 Multimodal Price Prediction Demo")

rows = []

n = len(text_data["price"])

for i in range(n):

    text_emb = text_data["text_emb"][i].unsqueeze(0).float()
    image_emb = image_data["image_emb"][i].unsqueeze(0).float()

    pred = predict(text_emb, image_emb)
    actual = text_data["price"][i].item()

    rows.append({
        "Product": text_data["clean_text"][i][:60],
        "Actual Price": actual,
        "Predicted Price": pred,
        "Error": abs(actual - pred)
    })

df = pd.DataFrame(rows)

# =========================
# TABLE VIEW
# =========================

st.subheader("📊 Prediction Table")
st.dataframe(df, use_container_width=True)

st.metric("📉 Average Error", f"{df['Error'].mean():.2f}")

# =========================
# PRODUCT VIEWER
# =========================

st.subheader("📦 Product Viewer")

selected = st.selectbox("Select Product", range(n))

col1, col2 = st.columns([1, 2])

# LEFT → IMAGE
with col1:

    img_path = os.path.join("train_images", text_data["image_name"][selected])

    if os.path.exists(img_path):
        st.image(Image.open(img_path), use_container_width=True)
    else:
        st.warning("Image not found")

# RIGHT → DETAILS
with col2:

    st.markdown("### 📝 Description")
    st.write(text_data["clean_text"][selected])

    actual = text_data["price"][selected].item()

    text_emb = text_data["text_emb"][selected].unsqueeze(0).float()
    image_emb = image_data["image_emb"][selected].unsqueeze(0).float()

    pred = predict(text_emb, image_emb)

    st.markdown("### 💰 Pricing")

    st.write(f"**Actual Price:** ₹{actual:.2f}")
    st.write(f"**Predicted Price:** ₹{pred:.2f}")
    st.write(f"**Error:** ₹{abs(actual - pred):.2f}")