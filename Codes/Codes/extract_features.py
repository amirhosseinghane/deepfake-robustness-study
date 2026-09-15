# -*- coding: utf-8 -*-
# Step 4: full feature extraction
# extract features for all images and save them to a file
# this step only needs to run once
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time

# --- fixed random seed ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- paths ---
MANIFEST = os.path.join(BASE_DIR, "manifest.csv")
OUTPUT   = os.path.join(BASE_DIR, "features.npz")

# ============================================================
# step 1: read the whole manifest
# ============================================================
df = pd.read_csv(MANIFEST)
print(f"Total images to process: {len(df)}")

# ============================================================
# step 2: set up the frozen model (like Step 3)
# ============================================================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Identity()
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ============================================================
# step 3: extract features with a progress print
# ============================================================
features = []
labels = []
failed = []  # if an image is broken, log it here

start_time = time.time()
total = len(df)

with torch.no_grad():
    for i, row in df.iterrows():
        try:
            img = Image.open(row["filepath"]).convert("RGB")
            tensor = preprocess(img).unsqueeze(0)
            feat = model(tensor).squeeze(0).numpy()

            features.append(feat)
            labels.append(row["label"])
        except Exception as e:
            # if an image fails to open, log it and skip
            failed.append((row["filepath"], str(e)))
            continue

        # --- simple progress: print every 100 images ---
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate
            print(f"  {i+1}/{total} images | "
                  f"elapsed {elapsed:.0f}s | "
                  f"~{remaining:.0f}s left")

# ============================================================
# step 4: save the result
# ============================================================
features = np.array(features)
labels = np.array(labels)

# save to a compressed file (features and labels together)
np.savez_compressed(OUTPUT, features=features, labels=labels)

total_time = time.time() - start_time
print(f"\n--- Done ---")
print(f"Feature matrix shape: {features.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Failed images: {len(failed)}")
print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"Saved to: {OUTPUT}")

# if any image was broken, show it
if failed:
    print("\nFailed files:")
    for path, err in failed[:10]:  # at most 10
        print(f"  {path} -> {err}")