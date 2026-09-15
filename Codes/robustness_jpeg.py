# Step 6 (part 1): robustness test - JPEG compression only
# the classifier is trained on clean data,
# but tested on the degraded test data
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

# --- fixed seed ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- paths ---
MANIFEST = os.path.join(BASE_DIR, "manifest.csv")
FEATURES = os.path.join(BASE_DIR, "features.npz")

# --- JPEG quality levels we want to test ---
# 100 = almost no loss, 10 = heavy compression
JPEG_QUALITIES = [95, 80, 60, 40, 20, 10]

# ============================================================
# step 1: reproduce the same train/test split
# ============================================================
# split on the manifest to get the test file paths
df = pd.read_csv(MANIFEST)

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"]
)
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# ============================================================
# step 2: train the classifier on clean training features
# ============================================================
# load the pre-extracted features
data = np.load(FEATURES)
X_all = data["features"]
y_all = data["labels"]

# features are in manifest order, so index them with train_df
X_train = X_all[train_df.index]
y_train = y_all[train_df.index]

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
print("Classifier trained on clean training features.")

# ============================================================
# step 3: set up the frozen model (to extract features of degraded images)
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

# --- degradation function: JPEG compression at a given quality ---
def jpeg_compress(img, quality):
    """compress the image as JPEG at the given quality and reopen it."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

# --- feature extraction function for a list of images ---
def extract_test_features(quality):
    """degrade the test images at the given quality and extract features.
       if quality=None, no degradation (for checking)."""
    feats = []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            img = Image.open(row["filepath"]).convert("RGB")
            if quality is not None:
                img = jpeg_compress(img, quality)   # degrade the raw image
            tensor = preprocess(img).unsqueeze(0)
            feat = model(tensor).squeeze(0).numpy()
            feats.append(feat)
    return np.array(feats)

# test labels (fixed, independent of degradation)
y_test = y_all[test_df.index]

# ============================================================
# step 4: evaluate at each quality level
# ============================================================
print("\n--- JPEG Robustness Test ---")
print(f"{'Quality':>8} | {'AUC':>7} | {'F1':>7}")
print("-" * 30)

results = []
for q in JPEG_QUALITIES:
    X_test_q = extract_test_features(q)
    y_prob = clf.predict_proba(X_test_q)[:, 1]
    y_pred = clf.predict(X_test_q)

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    results.append({"quality": q, "auc": auc, "f1": f1})

    print(f"{q:>8} | {auc:>7.4f} | {f1:>7.4f}")

# save results for plotting later
results_df = pd.DataFrame(results)
results_df.to_csv(
    os.path.join(BASE_DIR, "robustness_jpeg.csv"),
    index=False
)
print("\nResults saved.")
