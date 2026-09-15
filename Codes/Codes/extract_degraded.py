# -*- coding: utf-8 -*-
# Axis 2 - part 1: extract degraded features (train and test)
# one representative level per degradation. runs only once.
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import time

# --- fixed seed ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- paths ---
MANIFEST = os.path.join(BASE_DIR, "manifest.csv")
OUTPUT   = os.path.join(BASE_DIR, "degraded_features.npz")

# --- representative level for each degradation (from Axis 1 results) ---
NOISE_SIGMA = 30
JPEG_QUALITY = 20
RES_SCALE = 0.35

# ============================================================
# step 1: the usual train/test split
# ============================================================
df = pd.read_csv(MANIFEST)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"]
)
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# ============================================================
# step 2: frozen model and preprocessing
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
# step 3: degradation functions (same as Axis 1)
# ============================================================
def jpeg_compress(img, quality):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

def add_gaussian_noise(img, sigma):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy).convert("RGB")

def reduce_resolution(img, scale):
    w, h = img.size
    small = img.resize((max(1, int(w*scale)), max(1, int(h*scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR).convert("RGB")

# --- function that applies a given degradation to an image ---
def apply_degradation(img, kind):
    if kind == "clean":
        return img
    elif kind == "noise":
        return add_gaussian_noise(img, NOISE_SIGMA)
    elif kind == "jpeg":
        return jpeg_compress(img, JPEG_QUALITY)
    elif kind == "resolution":
        return reduce_resolution(img, RES_SCALE)

# ============================================================
# step 4: feature extraction for a dataframe with one degradation
# ============================================================
def extract(dataframe, kind):
    feats = []
    with torch.no_grad():
        for _, row in dataframe.iterrows():
            img = Image.open(row["filepath"]).convert("RGB")
            img = apply_degradation(img, kind)
            tensor = preprocess(img).unsqueeze(0)
            feat = model(tensor).squeeze(0).numpy()
            feats.append(feat)
    return np.array(feats)

# ============================================================
# step 5: extract for all conditions
# ============================================================
# only extract the three degradations; read clean from the previous file
degradations = ["noise", "jpeg", "resolution"]

storage = {}  # collect everything here

# labels (same for all conditions)
y_train = df.loc[train_df.index, "label"].values
y_test = df.loc[test_df.index, "label"].values
storage["y_train"] = y_train
storage["y_test"] = y_test

# --- read the clean condition from the Step 4 file ---
clean = np.load(os.path.join(BASE_DIR, "features.npz"))
X_all_clean = clean["features"]
storage["Xtrain_clean"] = X_all_clean[train_df.index]
storage["Xtest_clean"] = X_all_clean[test_df.index]
print("Loaded clean features from features.npz")

# --- extract the three degradations ---
for kind in degradations:
    t0 = time.time()
    print(f"\nExtracting '{kind}' ...")

    storage[f"Xtrain_{kind}"] = extract(train_df, kind)
    print(f"  train done ({time.time()-t0:.0f}s)")

    storage[f"Xtest_{kind}"] = extract(test_df, kind)
    print(f"  test done  ({time.time()-t0:.0f}s)")

# ============================================================
# step 6: save everything to one file
# ============================================================
np.savez_compressed(OUTPUT, **storage)
print(f"\nAll features saved to: {OUTPUT}")
print(f"Keys saved: {list(storage.keys())}")