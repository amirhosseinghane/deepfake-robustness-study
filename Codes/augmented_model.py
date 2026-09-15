# -*- coding: utf-8 -*-
# محور سوم: مدل با augmentation (ترکیب همه‌ی تخریب‌ها)
# از ویژگی‌های آماده در degraded_features.npz استفاده می‌کند
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEED = 42
np.random.seed(SEED)

DEGRADED = os.path.join(BASE_DIR, "degraded_features.npz")
BASE = BASE_DIR

# ============================================================
# گام ۱: بارگذاری همه‌ی ویژگی‌ها
# ============================================================
data = np.load(DEGRADED)
y_train = data["y_train"]
y_test = data["y_test"]

conditions = ["clean", "noise", "jpeg", "resolution"]

# ============================================================
# گام ۲: ساخت داده‌ی آموزشِ augmented
# ============================================================
# برای هر تصویر آموزش، یکی از چهار حالت را تصادفی انتخاب می‌کنیم.
# این‌طور اندازه‌ی آموزش ثابت می‌ماند ولی تنوع تخریب دارد.

n_train = len(y_train)

# برای هر نمونه، یک حالت تصادفی انتخاب کن (۰ تا ۳)
choice = np.random.randint(0, len(conditions), size=n_train)

# ماتریس ویژگی augmented را می‌سازیم
X_train_aug = np.zeros_like(data["Xtrain_clean"])
for idx, cond in enumerate(conditions):
    mask = (choice == idx)              # کدام نمونه‌ها این حالت را می‌گیرند
    X_train_aug[mask] = data[f"Xtrain_{cond}"][mask]

print(f"Augmented training set built: {X_train_aug.shape}")
print(f"Mix: {[f'{c}={np.sum(choice==i)}' for i, c in enumerate(conditions)]}")

# ============================================================
# گام ۳: آموزش مدل augmented
# ============================================================
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_aug, y_train)

# ============================================================
# گام ۴: سنجش روی همه‌ی حالت‌های تست
# ============================================================
print("\n=== Augmented Model — tested on each condition ===")
results = {}
for test_cond in conditions:
    X_test = data[f"Xtest_{test_cond}"]
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    results[test_cond] = auc
    print(f"  tested on {test_cond:>10}: AUC = {auc:.4f}")

# میانگین روی تخریب‌ها (برای مقایسه با محور دوم)
avg_degraded = np.mean([results[c] for c in ["noise", "jpeg", "resolution"]])
print(f"\nAvg AUC on degraded tests: {avg_degraded:.4f}")

# ============================================================
# گام ۵: مقایسه با بهترین مدل تک‌تخریبی (از محور دوم)
# ============================================================
print("\n--- Comparison ---")
print(f"{'Model':>20} | {'Avg on degraded':>16}")
print("-" * 42)
print(f"{'Trained on clean':>20} | {0.9410:>16.4f}")
print(f"{'Trained on noise':>20} | {0.9567:>16.4f}  (best single)")
print(f"{'Augmented (all)':>20} | {avg_degraded:>16.4f}  (this experiment)")

# ذخیره
pd.DataFrame([results]).to_csv(BASE + r"\augmented_results.csv", index=False)
print("\nSaved: augmented_results.csv")
