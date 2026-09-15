# -*- coding: utf-8 -*-
# محور دوم - تکه ۱: استخراج ویژگی‌های تخریب‌شده (آموزش و تست)
# برای هر تخریب، یک سطح نماینده. این اسکریپت فقط یک بار اجرا می‌شود.
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

# --- بذر ثابت ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- مسیرها ---
MANIFEST = os.path.join(BASE_DIR, "manifest.csv")
OUTPUT   = os.path.join(BASE_DIR, "degraded_features.npz")

# --- سطوح نماینده‌ی هر تخریب (از روی نتایج محور اول) ---
NOISE_SIGMA = 30
JPEG_QUALITY = 20
RES_SCALE = 0.35

# ============================================================
# گام ۱: همان تقسیم آموزش/آزمون همیشگی
# ============================================================
df = pd.read_csv(MANIFEST)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"]
)
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# ============================================================
# گام ۲: مدل منجمد و پیش‌پردازش
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
# گام ۳: توابع تخریب (همان‌های محور اول)
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

# --- تابعی که یک تخریب مشخص را روی یک تصویر اعمال می‌کند ---
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
# گام ۴: تابع استخراج ویژگی برای یک دیتافریم با یک تخریب
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
# گام ۵: استخراج برای همه‌ی حالت‌ها
# ============================================================
# فقط سه تخریب را استخراج می‌کنیم؛ سالم را از فایل قبلی می‌خوانیم
degradations = ["noise", "jpeg", "resolution"]

storage = {}  # اینجا همه‌چیز را جمع می‌کنیم

# برچسب‌ها (ثابت برای همه‌ی حالت‌ها)
y_train = df.loc[train_df.index, "label"].values
y_test = df.loc[test_df.index, "label"].values
storage["y_train"] = y_train
storage["y_test"] = y_test

# --- خواندن حالت سالم از فایل مرحله ۴ ---
clean = np.load(os.path.join(BASE_DIR, "features.npz"))
X_all_clean = clean["features"]
storage["Xtrain_clean"] = X_all_clean[train_df.index]
storage["Xtest_clean"] = X_all_clean[test_df.index]
print("Loaded clean features from features.npz")

# --- استخراج سه تخریب ---
for kind in degradations:
    t0 = time.time()
    print(f"\nExtracting '{kind}' ...")

    storage[f"Xtrain_{kind}"] = extract(train_df, kind)
    print(f"  train done ({time.time()-t0:.0f}s)")

    storage[f"Xtest_{kind}"] = extract(test_df, kind)
    print(f"  test done  ({time.time()-t0:.0f}s)")

# ============================================================
# گام ۶: ذخیره‌ی همه‌چیز در یک فایل
# ============================================================
np.savez_compressed(OUTPUT, **storage)
print(f"\nAll features saved to: {OUTPUT}")
print(f"Keys saved: {list(storage.keys())}")