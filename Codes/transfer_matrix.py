# -*- coding: utf-8 -*-
# محور دوم - تکه ۲: ساخت ماتریس انتقال
# برای هر ترکیب (آموزش با X، تست با Y) یک AUC محاسبه می‌کند
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEED = 42
np.random.seed(SEED)

# --- مسیر فایل ویژگی‌های تخریب‌شده ---
DEGRADED = os.path.join(BASE_DIR, "degraded_features.npz")
BASE = BASE_DIR

# ============================================================
# گام ۱: بارگذاری همه‌ی ویژگی‌ها
# ============================================================
data = np.load(DEGRADED)
y_train = data["y_train"]
y_test = data["y_test"]

# حالت‌های تخریب
conditions = ["clean", "noise", "jpeg", "resolution"]

# ============================================================
# گام ۲: محاسبه‌ی AUC برای هر ترکیب آموزش×تست
# ============================================================
# یک جدول خالی می‌سازیم: ردیف = آموزش، ستون = تست
matrix = pd.DataFrame(index=conditions, columns=conditions, dtype=float)

for train_cond in conditions:
    # ویژگی‌های آموزش با این تخریب
    X_train = data[f"Xtrain_{train_cond}"]

    # طبقه‌بند را یک بار آموزش می‌دهیم
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # حالا روی همه‌ی حالت‌های تست می‌سنجیم
    for test_cond in conditions:
        X_test = data[f"Xtest_{test_cond}"]
        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        matrix.loc[train_cond, test_cond] = auc

# ============================================================
# گام ۳: نمایش و ذخیره‌ی ماتریس
# ============================================================
print("\n=== Transfer Matrix (AUC) ===")
print("Rows = trained on | Columns = tested on\n")
print(matrix.round(4).to_string())

matrix.to_csv(BASE + r"\transfer_matrix.csv")
print("\nSaved: transfer_matrix.csv")

# ============================================================
# گام ۴: چند تحلیل خودکار برای کمک به تفسیر
# ============================================================
print("\n--- Quick analysis ---")

# میانگین عملکرد وقتی روی سالم آموزش دیده (ردیف clean)
print(f"Trained on CLEAN, avg on degraded tests: "
      f"{matrix.loc['clean', ['noise','jpeg','resolution']].mean():.4f}")

# آیا آموزش با تخریب، به‌طور کلی مقاومت را بهتر می‌کند؟
# میانگین هر ردیف روی تست‌های تخریب‌شده
for cond in conditions:
    avg_on_degraded = matrix.loc[cond, ['noise','jpeg','resolution']].mean()
    print(f"Trained on {cond:>10}: avg AUC on degraded tests = {avg_on_degraded:.4f}")