# -*- coding: utf-8 -*-
# اصلاحیه استاد: جدول کامل چهار متریک + چند seed
# از ویژگی‌های ذخیره‌شده استفاده می‌کند، پس سریع است
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score,
                             precision_score, recall_score)

# --- چند seed برای گزارش میانگین و انحراف معیار ---
SEEDS = [42, 7, 123, 2024, 99]

# --- مسیر فایل‌ها ---
FEATURES = os.path.join(BASE_DIR, "features.npz")
DEGRADED = os.path.join(BASE_DIR, "degraded_features.npz")

# ============================================================
# تابع: محاسبه‌ی چهار متریک برای یک حالت، روی چند seed
# ============================================================
def evaluate_multi_seed(X, y):
    """چهار متریک را روی چند seed محاسبه و میانگین±انحراف معیار برمی‌گرداند."""
    results = {"auc": [], "f1": [], "precision": [], "recall": []}
    for seed in SEEDS:
        np.random.seed(seed)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_te)[:, 1]
        y_pred = clf.predict(X_te)
        results["auc"].append(roc_auc_score(y_te, y_prob))
        results["f1"].append(f1_score(y_te, y_pred))
        results["precision"].append(precision_score(y_te, y_pred))
        results["recall"].append(recall_score(y_te, y_pred))
    # میانگین و انحراف معیار
    summary = {}
    for k, v in results.items():
        summary[k] = (np.mean(v), np.std(v))
    return summary

# ============================================================
# گام ۱: خط مبنا (داده سالم) با چند seed
# ============================================================
data = np.load(FEATURES)
X_clean, y_clean = data["features"], data["labels"]

print("="*65)
print("Baseline (clean data) — averaged over", len(SEEDS), "seeds")
print("="*65)
s = evaluate_multi_seed(X_clean, y_clean)
print(f"{'Metric':>12} | {'Mean':>8} | {'Std':>8}")
print("-"*35)
for metric in ["auc", "f1", "precision", "recall"]:
    mean, std = s[metric]
    print(f"{metric.upper():>12} | {mean:>8.4f} | {std:>8.4f}")

# ============================================================
# گام ۲: هر تخریب (از degraded_features) با چند seed
# ============================================================
# نکته: در degraded_features، تقسیم از قبل ثابت است،
# پس برای چند seed اینجا فقط طبقه‌بند را چند بار با
# حالت‌های تصادفی مختلف آموزش می‌دهیم تا پایداری را بسنجیم.
deg = np.load(DEGRADED)
y_test = deg["y_test"]
y_train = deg["y_train"]

conditions = {
    "clean": "Xtest_clean",
    "noise": "Xtest_noise",
    "jpeg": "Xtest_jpeg",
    "resolution": "Xtest_resolution",
}

print("\n" + "="*65)
print("Degraded test sets — classifier trained on clean train features")
print("="*65)
print(f"{'Condition':>12} | {'AUC':>7} | {'F1':>7} | {'Prec':>7} | {'Recall':>7}")
print("-"*55)

# طبقه‌بند روی ویژگی‌های سالمِ آموزش
X_train_clean = deg["Xtrain_clean"]
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_clean, y_train)

for name, key in conditions.items():
    X_te = deg[key]
    y_prob = clf.predict_proba(X_te)[:, 1]
    y_pred = clf.predict(X_te)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print(f"{name:>12} | {auc:>7.4f} | {f1:>7.4f} | {prec:>7.4f} | {rec:>7.4f}")

print("\nDone.")