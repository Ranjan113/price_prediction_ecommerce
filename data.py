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
    with Pool(16) as pool:
        list(tqdm(pool.imap_unordered(partial_download, image_links), total=len(image_links)))

if __name__ == "__main__":
    # Download training images
    # download_images_from_csv("/Users/ranjanmittal/Desktop/amazon/train.csv", "train_images")

    # Download test images
    download_images_from_csv("/Users/ranjanmittal/Desktop/amazon/test.csv", "test_images")

    # Download sample test images
    # download_images_from_csv("sample_test.csv", "sample_test_images")

    print("All downloads complete. You can now zip the folders for Kaggle upload.")