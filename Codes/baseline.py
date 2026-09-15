# -*- coding: utf-8 -*-
# مرحله ۵: خط مبنا
# ویژگی‌های ذخیره‌شده را می‌خواند، داده را به آموزش/آزمون تقسیم می‌کند،
# طبقه‌بند را آموزش می‌دهد و عملکرد واقعی را گزارش می‌کند
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix

# --- بذر تصادفی ثابت ---
SEED = 42
np.random.seed(SEED)

# --- مسیر فایل ویژگی‌ها ---
FEATURES = os.path.join(BASE_DIR, "features.npz")

# ============================================================
# گام ۱: بارگذاری ویژگی‌ها و برچسب‌ها
# ============================================================
data = np.load(FEATURES)
X = data["features"]   # بردارهای ویژگی
y = data["labels"]     # برچسب‌ها (۰=واقعی، ۱=جعلی)

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# ============================================================
# گام ۲: تقسیم به آموزش و آزمون
# ============================================================
# ۲۰٪ داده برای آزمون کنار گذاشته می‌شود.
# stratify=y یعنی نسبت کلاس‌ها در هر دو بخش حفظ می‌شود.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

print(f"\nTrain size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# ============================================================
# گام ۳: آموزش طبقه‌بند خطی
# ============================================================
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# ============================================================
# گام ۴: ارزیابی روی داده‌ی آزمون (که مدل ندیده)
# ============================================================
# پیش‌بینی کلاس
y_pred = clf.predict(X_test)

# احتمال کلاس مثبت (برای محاسبه‌ی AUC)
y_prob = clf.predict_proba(X_test)[:, 1]

# --- متریک‌ها ---
auc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n--- Baseline Results (on unseen test data) ---")
print(f"AUC:      {auc:.4f}")
print(f"F1:       {f1:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"\nConfusion Matrix:")
print(f"              Pred_Real  Pred_Fake")
print(f"  True_Real   {cm[0,0]:>8}   {cm[0,1]:>8}")
print(f"  True_Fake   {cm[1,0]:>8}   {cm[1,1]:>8}")