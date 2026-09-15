# -*- coding: utf-8 -*-
# مرحله ۶ (بخش دوم): آزمون مقاومت — نویز گوسی
# ساختار دقیقاً مثل فایل JPEG است، فقط تابع تخریب عوض شده
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

# --- بذر ثابت ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- مسیرها ---
MANIFEST = os.path.join(BASE_DIR, "manifest.csv")
FEATURES = os.path.join(BASE_DIR, "features.npz")

# --- سطوح نویز (سیگما) ---
# ۰ = بدون نویز، ۴۰ = نویز شدید
NOISE_SIGMAS = [5, 10, 20, 30, 40]

# ============================================================
# گام ۱: بازتولید همان تقسیم آموزش/آزمون
# ============================================================
df = pd.read_csv(MANIFEST)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"]
)
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# ============================================================
# گام ۲: آموزش طبقه‌بند روی ویژگی‌های سالمِ آموزش
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
# گام ۳: مدل منجمد و پیش‌پردازش
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

# --- تابع تخریب: افزودن نویز گوسی ---
def add_gaussian_noise(img, sigma):
    """نویز تصادفی گوسی با انحراف معیار sigma به تصویر اضافه می‌کند."""
    arr = np.array(img).astype(np.float32)          # تصویر به آرایه‌ی عددی
    noise = np.random.normal(0, sigma, arr.shape)   # نویز تصادفی
    noisy = arr + noise                             # افزودن نویز
    noisy = np.clip(noisy, 0, 255).astype(np.uint8) # محدود کردن به بازه‌ی معتبر
    return Image.fromarray(noisy).convert("RGB")    # برگرداندن به تصویر

# --- استخراج ویژگی تصاویر آزمونِ تخریب‌شده ---
def extract_test_features(sigma):
    feats = []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            img = Image.open(row["filepath"]).convert("RGB")
            img = add_gaussian_noise(img, sigma)   # تخریب روی تصویر خام
            tensor = preprocess(img).unsqueeze(0)
            feat = model(tensor).squeeze(0).numpy()
            feats.append(feat)
    return np.array(feats)

y_test = y_all[test_df.index]

# ============================================================
# گام ۴: سنجش در هر سطح نویز
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