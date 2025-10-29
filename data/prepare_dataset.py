"""
Prepare NIH Chest X-ray14 dataset
---------------------------------
Download, extract, and preprocess images.
Resize and split into train/test sets.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

DATA_DIR = "data/raw_images"
OUTPUT_DIR = "data/processed"
CSV_PATH = "data/dataset.csv"

def prepare_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv("\path\to\chest_xray14_labels.csv")
    df["image_path"] = df["Image Index"].apply(lambda x: os.path.join(DATA_DIR, x))
    
    # Split into train/test (80/20)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)
    print("Dataset prepared and split!")

if __name__ == "__main__":
    prepare_dataset()
