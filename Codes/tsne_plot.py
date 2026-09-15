# -*- coding: utf-8 -*-
# اصلاحیه استاد: نمودار t-SNE از فضای ویژگی
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

FEATURES = os.path.join(BASE_DIR, "features.npz")
BASE = BASE_DIR

SEED = 42
np.random.seed(SEED)

# بارگذاری ویژگی‌ها
data = np.load(FEATURES)
X, y = data["features"], data["labels"]

# --- برای سرعت، یک زیرنمونه‌ی تصادفی می‌گیریم ---
# t-SNE روی ۳۲۰۰ نقطه کند است؛ ۱۰۰۰ نقطه کافی و سریع‌تر است
n_sample = 1000
idx = np.random.choice(len(X), size=n_sample, replace=False)
X_sample = X[idx]
y_sample = y[idx]

print("Running t-SNE... (این ممکن است ۱ تا ۳ دقیقه طول بکشد)")

# --- اجرای t-SNE: کاهش از ۵۱۲ بعد به ۲ بعد ---
tsne = TSNE(n_components=2, random_state=SEED,
            perplexity=30, init="pca")
X_2d = tsne.fit_transform(X_sample)

# --- رسم ---
plt.figure(figsize=(8, 7))

# نقاط واقعی (برچسب ۰) و جعلی (برچسب ۱) با رنگ متفاوت
real_mask = (y_sample == 0)
fake_mask = (y_sample == 1)

plt.scatter(X_2d[real_mask, 0], X_2d[real_mask, 1],
            c="tab:blue", label="Real", alpha=0.6, s=20)
plt.scatter(X_2d[fake_mask, 0], X_2d[fake_mask, 1],
            c="tab:red", label="Fake", alpha=0.6, s=20)

plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.title("t-SNE Visualization of Feature Space (Clean Data)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(BASE + r"\fig_tsne.png", dpi=150, bbox_inches="tight")
print("Saved: fig_tsne.png")
print("Done.")