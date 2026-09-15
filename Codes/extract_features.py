# -*- coding: utf-8 -*-
# مرحله ۴: استخراج ویژگی کامل
# ویژگی همه‌ی تصاویر را استخراج و در یک فایل ذخیره می‌کند
# این مرحله فقط یک بار اجرا می‌شود
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time

# --- بذر تصادفی ثابت ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- مسیرها ---
MANIFEST = os.path.join(BASE_DIR, "manifest.csv")
OUTPUT   = os.path.join(BASE_DIR, "features.npz")

# ============================================================
# گام ۱: خواندن کل مانیفست
# ============================================================
df = pd.read_csv(MANIFEST)
print(f"Total images to process: {len(df)}")

# ============================================================
# گام ۲: آماده‌سازی مدل منجمد (مثل مرحله ۳)
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
# گام ۳: استخراج ویژگی با نوار پیشرفت
# ============================================================
features = []
labels = []
failed = []  # اگر تصویری خراب بود، اینجا ثبت می‌شود

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
            # اگر تصویری باز نشد، ثبتش می‌کنیم و رد می‌شویم
            failed.append((row["filepath"], str(e)))
            continue

        # --- نوار پیشرفت ساده: هر ۱۰۰ تصویر یک گزارش ---
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate
            print(f"  {i+1}/{total} images | "
                  f"elapsed {elapsed:.0f}s | "
                  f"~{remaining:.0f}s left")

# ============================================================
# گام ۴: ذخیره‌ی نتیجه
# ============================================================
features = np.array(features)
labels = np.array(labels)

# ذخیره در یک فایل فشرده (هم بردارها هم برچسب‌ها با هم)
np.savez_compressed(OUTPUT, features=features, labels=labels)

total_time = time.time() - start_time
print(f"\n--- Done ---")
print(f"Feature matrix shape: {features.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Failed images: {len(failed)}")
print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"Saved to: {OUTPUT}")

# اگر تصویری خراب بود، نشانش بده
if failed:
    print("\nFailed files:")
    for path, err in failed[:10]:  # حداکثر ۱۰ تا
        print(f"  {path} -> {err}")