import os
import cv2
import pandas as pd
import easyocr
from tqdm import tqdm
from multiprocessing import cpu_count, get_context

# ================= CONFIG =================
IMAGE_DIR = "train_images"
CSV_PATH = "train.csv"
OUTPUT_CSV = "easyocr_train_cache.csv"

LANG = ['en']
USE_GPU = False              # CPU only (safe for Mac)
RESIZE = True
SAVE_EVERY = 1000            # Save every 1000 images
# =========================================


# ---------- OCR WORKER ----------
def ocr_worker(args):
    image_links, image_dir = args

    reader = easyocr.Reader(LANG, gpu=USE_GPU)
    results = []

    for link in image_links:
        img_name = str(link).split("/")[-1]
        img_path = os.path.join(image_dir, img_name)

        if not os.path.exists(img_path):
            results.append((img_path, ""))
            continue

        img = cv2.imread(img_path)
        if img is None:
            results.append((img_path, ""))
            continue

        if RESIZE:
            img = cv2.resize(img, (640, 640))

        ocr = reader.readtext(img, paragraph=False)
        text = " ".join([r[1] for r in ocr if r[2] > 0.4])

        results.append((img_path, text))

    return results


# ---------- MAIN ----------
if __name__ == "__main__":

    df = pd.read_csv(CSV_PATH)
    image_links = df["image_link"].tolist()

    # ---- LOAD EXISTING PROGRESS ----
    if os.path.exists(OUTPUT_CSV):
        cached_df = pd.read_csv(OUTPUT_CSV)
        done_paths = set(cached_df["image_path"])
        print(f"🔁 Resuming: {len(done_paths)} images already processed")
    else:
        cached_df = pd.DataFrame(columns=["image_path", "ocr_text"])
        done_paths = set()
        print("🆕 Starting fresh OCR run")

    # ---- FILTER REMAINING IMAGES ----
    remaining_links = []
    for link in image_links:
        img_name = str(link).split("/")[-1]
        img_path = os.path.join(IMAGE_DIR, img_name)
        if img_path not in done_paths:
            remaining_links.append(link)

    if not remaining_links:
        print("✅ All images already processed!")
        exit()

    print(f"📸 Remaining images: {len(remaining_links)}")

    # ---- MULTIPROCESS SETUP ----
    NUM_WORKERS = max(cpu_count() - 1, 1)
    print(f"⚙️ Using {NUM_WORKERS} CPU cores")

    ctx = get_context("spawn")

    # ---- PROCESS IN SAVE_EVERY BATCHES ----
    for start in range(0, len(remaining_links), SAVE_EVERY):
        batch = remaining_links[start:start + SAVE_EVERY]

        print(f"\n🚀 Processing images {start + 1} → {start + len(batch)}")

        # Evenly distribute work
        chunks = [[] for _ in range(NUM_WORKERS)]
        for i, link in enumerate(batch):
            chunks[i % NUM_WORKERS].append(link)

        chunks = [c for c in chunks if c]

        batch_results = []

        with ctx.Pool(
            processes=len(chunks),
            maxtasksperchild=1
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(
                    ocr_worker,
                    [(chunk, IMAGE_DIR) for chunk in chunks]
                ),
                total=len(chunks),
                desc="EasyOCR workers"
            ):
                batch_results.extend(result)

        batch_df = pd.DataFrame(batch_results, columns=["image_path", "ocr_text"])
        cached_df = pd.concat([cached_df, batch_df], ignore_index=True)

        cached_df.to_csv(OUTPUT_CSV, index=False)
        print(f"💾 Saved progress → {OUTPUT_CSV}")

    print("\n✅ OCR COMPLETE — Safe to shut down Mac")
