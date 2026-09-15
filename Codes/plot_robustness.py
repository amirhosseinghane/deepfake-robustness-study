# -*- coding: utf-8 -*-
# رسم نمودارهای مقاومت از روی سه فایل CSV
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import pandas as pd
import matplotlib.pyplot as plt

# --- مسیر پوشه ---
BASE = BASE_DIR

# --- خط مبنا (بدون تخریب) از مرحله ۵ ---
BASELINE_AUC = 0.9826
BASELINE_F1 = 0.9444

# --- خواندن سه فایل نتایج ---
jpeg = pd.read_csv(BASE + r"\robustness_jpeg.csv")
noise = pd.read_csv(BASE + r"\robustness_noise.csv")
res = pd.read_csv(BASE + r"\robustness_resolution.csv")

# ============================================================
# نمودار ۱: سه تخریب جداگانه، کنار هم
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# --- JPEG ---
axes[0].plot(jpeg["quality"], jpeg["auc"], "o-", label="AUC", color="tab:blue")
axes[0].plot(jpeg["quality"], jpeg["f1"], "s--", label="F1", color="tab:orange")
axes[0].axhline(BASELINE_AUC, color="gray", linestyle=":", alpha=0.7)
axes[0].set_title("JPEG Compression")
axes[0].set_xlabel("Quality (lower = more compression)")
axes[0].set_ylabel("Score")
axes[0].invert_xaxis()  # کیفیت پایین سمت راست باشد (بدتر شدن به راست)
axes[0].legend()
axes[0].grid(alpha=0.3)

# --- Noise ---
axes[1].plot(noise["sigma"], noise["auc"], "o-", label="AUC", color="tab:blue")
axes[1].plot(noise["sigma"], noise["f1"], "s--", label="F1", color="tab:orange")
axes[1].axhline(BASELINE_AUC, color="gray", linestyle=":", alpha=0.7)
axes[1].set_title("Gaussian Noise")
axes[1].set_xlabel("Sigma (higher = more noise)")
axes[1].legend()
axes[1].grid(alpha=0.3)

# --- Resolution ---
axes[2].plot(res["scale"], res["auc"], "o-", label="AUC", color="tab:blue")
axes[2].plot(res["scale"], res["f1"], "s--", label="F1", color="tab:orange")
axes[2].axhline(BASELINE_AUC, color="gray", linestyle=":", alpha=0.7)
axes[2].set_title("Resolution Reduction")
axes[2].set_xlabel("Scale (lower = less resolution)")
axes[2].invert_xaxis()  # scale پایین سمت راست باشد
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.suptitle("Robustness of Frozen ResNet-18 Features to Image Degradations",
             fontsize=13)
plt.tight_layout()
plt.savefig(BASE + r"\fig_robustness_separate.png", dpi=150, bbox_inches="tight")
print("Saved: fig_robustness_separate.png")

# ============================================================
# نمودار ۲: مقایسه‌ی سه تخریب روی محور مشترک «شدت»
# ============================================================
# برای هر تخریب، یک «شدت نرمال‌شده» از ۰ تا ۱ می‌سازیم
# ۰ = ملایم‌ترین، ۱ = شدیدترین

fig2, ax = plt.subplots(figsize=(8, 5))

# JPEG: کیفیت بالا = شدت کم. نرمال‌سازی معکوس.
jpeg_severity = 1 - (jpeg["quality"] - jpeg["quality"].min()) / \
                    (jpeg["quality"].max() - jpeg["quality"].min())
ax.plot(jpeg_severity, jpeg["auc"], "o-", label="JPEG Compression")

# Noise: سیگما بالا = شدت زیاد. نرمال‌سازی مستقیم.
noise_severity = (noise["sigma"] - noise["sigma"].min()) / \
                 (noise["sigma"].max() - noise["sigma"].min())
ax.plot(noise_severity, noise["auc"], "s-", label="Gaussian Noise")

# Resolution: scale بالا = شدت کم. نرمال‌سازی معکوس.
res_severity = 1 - (res["scale"] - res["scale"].min()) / \
                   (res["scale"].max() - res["scale"].min())
ax.plot(res_severity, res["auc"], "^-", label="Resolution Reduction")

ax.axhline(BASELINE_AUC, color="gray", linestyle=":", label="Baseline (no degradation)")
ax.set_title("AUC vs Degradation Severity (comparison)")
ax.set_xlabel("Degradation Severity (0 = mild, 1 = severe)")
ax.set_ylabel("AUC")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(BASE + r"\fig_robustness_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: fig_robustness_comparison.png")

print("\nDone! Two figures saved in Codes folder.")