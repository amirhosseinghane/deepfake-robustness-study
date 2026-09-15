# -*- coding: utf-8 -*-
# رسم heatmap از ماتریس انتقال
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = BASE_DIR

# --- خواندن ماتریس ---
matrix = pd.read_csv(BASE + r"\transfer_matrix.csv", index_col=0)
conditions = list(matrix.index)
values = matrix.values.astype(float)

# ============================================================
# رسم heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))

# نقشه‌ی رنگ: هرچه AUC بالاتر، روشن‌تر
im = ax.imshow(values, cmap="viridis", vmin=0.85, vmax=1.0, aspect="auto")

# --- نوار رنگ کنار نمودار ---
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("AUC", rotation=270, labelpad=15)

# --- برچسب محورها ---
ax.set_xticks(range(len(conditions)))
ax.set_yticks(range(len(conditions)))
ax.set_xticklabels(conditions)
ax.set_yticklabels(conditions)
ax.set_xlabel("Tested on", fontsize=11)
ax.set_ylabel("Trained on", fontsize=11)
ax.set_title("Cross-Degradation Transfer Matrix (AUC)", fontsize=12, pad=12)

# --- نوشتن عدد داخل هر خانه ---
for i in range(len(conditions)):
    for j in range(len(conditions)):
        val = values[i, j]
        # رنگ متن: روی خانه‌های تیره سفید، روی روشن مشکی
        color = "white" if val < 0.93 else "black"
        # قطر اصلی را پررنگ‌تر نشان می‌دهیم
        weight = "bold" if i == j else "normal"
        ax.text(j, i, f"{val:.3f}",
                ha="center", va="center",
                color=color, fontweight=weight, fontsize=10)

# --- کادر دور خانه‌های قطر اصلی (که مدل همان تخریب را دیده) ---
for i in range(len(conditions)):
    ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1,
                               fill=False, edgecolor="red",
                               linewidth=2))

plt.tight_layout()
plt.savefig(BASE + r"\fig_transfer_matrix.png", dpi=150, bbox_inches="tight")
print("Saved: fig_transfer_matrix.png")
print("Done!")