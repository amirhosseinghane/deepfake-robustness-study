# -*- coding: utf-8 -*-
# مرحله ۶ (بخش سوم): آزمون مقاومت — کاهش رزولوشن
# ساختار مثل دو فایل قبل، فقط تابع تخریب عوض شده
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

# --- ضرایب کاهش رزولوشن ---
# ۰.۷۵ = کاهش ملایم، ۰.۲۵ = کاهش شدید
SCALES = [0.75, 0.5, 0.35, 0.25]

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

# --- تابع تخریب: کاهش و بازگردانی رزولوشن ---
def reduce_resolution(img, scale):
    """تصویر را با ضریب scale کوچک و دوباره به اندازه‌ی اصلی بزرگ می‌کند."""
    w, h = img.size                              # اندازه‌ی اصلی
    small_w = max(1, int(w * scale))             # عرض کوچک‌شده
    small_h = max(1, int(h * scale))             # ارتفاع کوچک‌شده
    # کوچک کردن (اطلاعات از دست می‌رود)
    small = img.resize((small_w, small_h), Image.BILINEAR)
    # بزرگ کردن دوباره به اندازه‌ی اصلی
    restored = small.resize((w, h), Image.BILINEAR)
    return restored.convert("RGB")

# --- استخراج ویژگی تصاویر آزمونِ تخریب‌شده ---
def extract_test_features(scale):
    feats = []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            img = Image.open(row["filepath"]).convert("RGB")
            img = reduce_resolution(img, scale)   # تخریب روی تصویر خام
            tensor = preprocess(img).unsqueeze(0)
            feat = model(tensor).squeeze(0).numpy()
            feats.append(feat)
    return np.array(feats)

y_test = y_all[test_df.index]

# ============================================================
# گام ۴: سنجش در هر سطح رزولوشن
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