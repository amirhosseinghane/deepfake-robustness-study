# Step 6 (part 3): robustness test - resolution reduction
# same structure as the previous two files, only the degradation function changes
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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

# --- resolution reduction factors ---
# 0.75 = mild reduction, 0.25 = heavy reduction
SCALES = [0.75, 0.5, 0.35, 0.25]

# ============================================================
# step 1: reproduce the same train/test split
# ============================================================
df = pd.read_csv(MANIFEST)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"]
)
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# ============================================================
# step 2: train the classifier on clean training features
# ============================================================
data = np.load(FEATURES)
X_all = data["features"]
y_all = data["labels"]

X_train = X_all[train_df.index]
y_train = y_all[train_df.index]

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
print("Classifier trained on clean training features.")

# ============================================================
# step 3: frozen model and preprocessing
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

# --- degradation function: reduce and restore resolution ---
def reduce_resolution(img, scale):
    """shrink the image by scale then upscale back to the original size."""
    w, h = img.size                              # original size
    small_w = max(1, int(w * scale))             # reduced width
    small_h = max(1, int(h * scale))             # reduced height
    # shrink (information is lost)
    small = img.resize((small_w, small_h), Image.BILINEAR)
    # upscale back to the original size
    restored = small.resize((w, h), Image.BILINEAR)
    return restored.convert("RGB")

# --- extract features of the degraded test images ---
def extract_test_features(scale):
    feats = []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            img = Image.open(row["filepath"]).convert("RGB")
            img = reduce_resolution(img, scale)   # degrade the raw image
            tensor = preprocess(img).unsqueeze(0)
            feat = model(tensor).squeeze(0).numpy()
            feats.append(feat)
    return np.array(feats)

y_test = y_all[test_df.index]

# ============================================================
# step 4: evaluate at each resolution level
# ============================================================
print("\n--- Resolution Reduction Robustness Test ---")
print(f"{'Scale':>6} | {'AUC':>7} | {'F1':>7}")
print("-" * 28)

results = []
for sc in SCALES:
    X_test_sc = extract_test_features(sc)
    y_prob = clf.predict_proba(X_test_sc)[:, 1]
    y_pred = clf.predict(X_test_sc)

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    results.append({"scale": sc, "auc": auc, "f1": f1})

    print(f"{sc:>6} | {auc:>7.4f} | {f1:>7.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv(
    os.path.join(BASE_DIR, "robustness_resolution.csv"),
    index=False
)
print("\nResults saved.")
