import torch
from sentence_transformers import SentenceTransformer
import open_clip
from PIL import Image

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

# Load text model
print("\nLoading MiniLM...")
text_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Text model loaded!")

# Load image model
print("\nLoading EVA02-L-14...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="EVA02-L-14",
    pretrained="merged2b_s4b_b131k",
    device=device
)
clip_model.eval()
print("Image model loaded!")

# Test embeddings
text_emb = text_model.encode("Nescafe coffee 100g")
print("\nText embedding shape:", text_emb.shape)

img = Image.new("RGB", (224, 224))
img_tensor = preprocess(img).unsqueeze(0).to(device)

with torch.no_grad():
    img_emb = clip_model.encode_image(img_tensor)

print("Image embedding shape:", img_emb.shape)

print("\n✅ Everything working properly!")