# -*- coding: utf-8 -*-
# professor feedback: ROC curve for each degradation
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

DEGRADED = os.path.join(BASE_DIR, "degraded_features.npz")
BASE = BASE_DIR

# load
deg = np.load(DEGRADED)
y_train = deg["y_train"]
y_test = deg["y_test"]

# classifier on clean training features
clf = LogisticRegression(max_iter=1000)
clf.fit(deg["Xtrain_clean"], y_train)

# test conditions
conditions = {
    "Clean": "Xtest_clean",
    "JPEG (Q=20)": "Xtest_jpeg",
    "Noise (σ=30)": "Xtest_noise",
    "Resolution (×0.35)": "Xtest_resolution",
}

# --- plot the ROC curve for each condition ---
plt.figure(figsize=(7, 6))

for name, key in conditions.items():
    X_te = deg[key]
    y_prob = clf.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

# diagonal line (random guess)
plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves under Different Degradations")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(BASE + r"\fig_roc_curves.png", dpi=150, bbox_inches="tight")
print("Saved: fig_roc_curves.png")
print("Done.")