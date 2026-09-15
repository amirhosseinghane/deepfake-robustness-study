# -*- coding: utf-8 -*-
# اصلاحیه استاد: تحلیل خطای عمیق
# بررسی می‌کند مدل روی کدام تصاویر بیشتر اشتباه می‌کند
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

FEATURES = os.path.join(BASE_DIR, "features.npz")
MANIFEST = os.path.join(BASE_DIR, "manifest.csv")
BASE = BASE_DIR

# بارگذاری ویژگی‌ها و مانیفست (برای مسیر فایل‌ها)
data = np.load(FEATURES)
X, y = data["features"], data["labels"]
df = pd.read_csv(MANIFEST)

# همان تقسیم همیشگی
indices = np.arange(len(X))
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X, y, indices, test_size=0.2, random_state=SEED, stratify=y)

# آموزش و پیش‌بینی
clf = LogisticRegression(max_iter=1000)
clf.fit(X_tr, y_tr)
y_prob = clf.predict_proba(X_te)[:, 1]
y_pred = clf.predict(X_te)

# ============================================================
# گام ۱: پیدا کردن خطاها
# ============================================================
errors = (y_pred != y_te)
n_errors = errors.sum()
print(f"Total test samples: {len(y_te)}")
print(f"Total errors: {n_errors} ({100*n_errors/len(y_te):.1f}%)")

# تفکیک دو نوع خطا
false_positives = (y_pred == 1) & (y_te == 0)  # واقعی که جعلی خوانده شد
false_negatives = (y_pred == 0) & (y_te == 1)  # جعلی که واقعی خوانده شد
print(f"  False Positives (real called fake): {false_positives.sum()}")
print(f"  False Negatives (fake called real): {false_negatives.sum()}")

# ============================================================
# گام ۲: تحلیل «اطمینان» مدل در خطاها
# ============================================================
# آیا مدل در خطاهایش مطمئن بوده یا مردد؟
# فاصله احتمال از ۰.۵ = میزان اطمینان
confidence = np.abs(y_prob - 0.5)

error_conf = confidence[errors].mean()
correct_conf = confidence[~errors].mean()
print(f"\nMean confidence on correct predictions: {correct_conf:.3f}")
print(f"Mean confidence on errors:              {error_conf:.3f}")
print("(کمتر بودن اطمینان در خطاها یعنی مدل در اشتباهاتش مردد بوده)")

# ============================================================
# گام ۳: سخت‌ترین نمونه‌ها (مطمئن‌ترین خطاها)
# ============================================================
# خطاهایی که مدل بیشترین اطمینان را داشته ولی اشتباه کرده
error_indices = np.where(errors)[0]
error_confidences = confidence[error_indices]
# مرتب‌سازی بر اساس اطمینان (نزولی)
hardest = error_indices[np.argsort(-error_confidences)][:10]

print(f"\n--- 10 hardest errors (most confident mistakes) ---")
print(f"{'File':>40} | {'True':>5} | {'Pred':>5} | {'Prob':>6}")
print("-"*65)
for i in hardest:
    global_idx = idx_te[i]
    filepath = df.iloc[global_idx]["filepath"]
    filename = filepath.split("\\")[-1][:38]  # فقط نام فایل
    true_label = "fake" if y_te[i] == 1 else "real"
    pred_label = "fake" if y_pred[i] == 1 else "real"
    print(f"{filename:>40} | {true_label:>5} | {pred_label:>5} | {y_prob[i]:>6.3f}")

# ============================================================
# گام ۴: ذخیره‌ی خلاصه برای گزارش
# ============================================================
summary = {
    "total_test": len(y_te),
    "total_errors": int(n_errors),
    "false_positives": int(false_positives.sum()),
    "false_negatives": int(false_negatives.sum()),
    "correct_confidence": round(correct_conf, 3),
    "error_confidence": round(error_conf, 3),
}
print(f"\n--- Summary ---")
for k, v in summary.items():
    print(f"  {k}: {v}")

print("\nDone.")