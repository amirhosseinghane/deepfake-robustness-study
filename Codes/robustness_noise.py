# Step 6 (part 2): robustness test - Gaussian noise
# same structure as the JPEG file, only the degradation function changes
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

# --- noise levels (sigma) ---
# 0 = no noise, 40 = heavy noise
NOISE_SIGMAS = [5, 10, 20, 30, 40]

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

# --- degradation function: add Gaussian noise ---
def add_gaussian_noise(img, sigma):
    """add random Gaussian noise with std sigma to the image."""
    arr = np.array(img).astype(np.float32)          # image to numeric array
    noise = np.random.normal(0, sigma, arr.shape)   # random noise
    noisy = arr + noise                             # add noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8) # clip to the valid range
    return Image.fromarray(noisy).convert("RGB")    # convert back to image

# --- extract features of the degraded test images ---
def extract_test_features(sigma):
    feats = []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            img = Image.open(row["filepath"]).convert("RGB")
            img = add_gaussian_noise(img, sigma)   # degrade the raw image
            tensor = preprocess(img).unsqueeze(0)
            feat = model(tensor).squeeze(0).numpy()
            feats.append(feat)
    return np.array(feats)

y_test = y_all[test_df.index]

# ============================================================
# step 4: evaluate at each noise level
# ============================================================
print("\n--- Gaussian Noise Robustness Test ---")
print(f"{'Sigma':>6} | {'AUC':>7} | {'F1':>7}")
print("-" * 28)

results = []
for s in NOISE_SIGMAS:
    X_test_s = extract_test_features(s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]
    y_pred = clf.predict(X_test_s)

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    results.append({"sigma": s, "auc": auc, "f1": f1})

    print(f"{s:>6} | {auc:>7.4f} | {f1:>7.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv(
    os.path.join(BASE_DIR, "robustness_noise.csv"),
    index=False
)
print("\nResults saved.")
