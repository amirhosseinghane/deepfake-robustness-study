# -*- coding: utf-8 -*-
# مرحله ۶ (بخش اول): آزمون مقاومت — فقط فشرده‌سازی JPEG
# طبقه‌بند روی داده‌ی سالم آموزش می‌بیند،
# ولی روی داده‌ی آزمونِ تخریب‌شده سنجیده می‌شود
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

# --- بذر ثابت ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- مسیرها ---
MANIFEST = os.path.join(BASE_DIR, "manifest.csv")
FEATURES = os.path.join(BASE_DIR, "features.npz")

# --- سطوح کیفیت JPEG که می‌خواهیم آزمایش کنیم ---
# ۱۰۰ = تقریباً بدون افت، ۱۰ = فشرده‌سازی شدید
JPEG_QUALITIES = [95, 80, 60, 40, 20, 10]

# ============================================================
# گام ۱: بازتولید همان تقسیم آموزش/آزمون
# ============================================================
# روی مانیفست تقسیم می‌کنیم تا مسیر فایل‌های آزمون را داشته باشیم
df = pd.read_csv(MANIFEST)

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"]
)
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# ============================================================
# گام ۲: آموزش طبقه‌بند روی ویژگی‌های سالمِ آموزش
# ============================================================
# ویژگی‌های از پیش استخراج‌شده را بارگذاری می‌کنیم
data = np.load(FEATURES)
X_all = data["features"]
y_all = data["labels"]

# ویژگی‌ها به همان ترتیب مانیفست‌اند، پس با اندیس train_df جدا می‌کنیم
X_train = X_all[train_df.index]
y_train = y_all[train_df.index]

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
print("Classifier trained on clean training features.")

# ============================================================
# گام ۳: آماده‌سازی مدل منجمد (برای استخراج ویژگیِ تصاویر تخریب‌شده)
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

# --- تابع تخریب: فشرده‌سازی JPEG با کیفیت مشخص ---
def jpeg_compress(img, quality):
    """تصویر را با کیفیت داده‌شده JPEG فشرده و دوباره باز می‌کند."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

# --- تابع استخراج ویژگی برای یک لیست از تصاویر ---
def extract_test_features(quality):
    """تصاویر آزمون را با کیفیت داده‌شده تخریب و ویژگی استخراج می‌کند.
       اگر quality=None باشد، بدون تخریب (برای بررسی)."""
    feats = []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            img = Image.open(row["filepath"]).convert("RGB")
            if quality is not None:
                img = jpeg_compress(img, quality)   # تخریب روی تصویر خام
            tensor = preprocess(img).unsqueeze(0)
            feat = model(tensor).squeeze(0).numpy()
            feats.append(feat)
    return np.array(feats)

# برچسب‌های آزمون (ثابت، مستقل از تخریب)
y_test = y_all[test_df.index]

# ============================================================
# گام ۴: سنجش در هر سطح کیفیت
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

# ذخیره‌ی نتایج برای رسم نمودار بعداً
results_df = pd.DataFrame(results)
results_df.to_csv(
    os.path.join(BASE_DIR, "robustness_jpeg.csv"),
    index=False
)
print("\nResults saved.")