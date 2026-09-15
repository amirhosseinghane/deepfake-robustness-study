# -*- coding: utf-8 -*-
# professor feedback: t-SNE plot of the feature space
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

FEATURES = os.path.join(BASE_DIR, "features.npz")
BASE = BASE_DIR

SEED = 42
np.random.seed(SEED)

# load features
data = np.load(FEATURES)
X, y = data["features"], data["labels"]

# --- take a random subsample for speed ---
# t-SNE is slow on 3200 points; 1000 points is enough and faster
n_sample = 1000
idx = np.random.choice(len(X), size=n_sample, replace=False)
X_sample = X[idx]
y_sample = y[idx]

print("Running t-SNE... (this may take 1 to 3 minutes)")

# --- run t-SNE: reduce from 512 dims to 2 dims ---
tsne = TSNE(n_components=2, random_state=SEED,
            perplexity=30, init="pca")
X_2d = tsne.fit_transform(X_sample)

# --- plot ---
plt.figure(figsize=(8, 7))

# real points (label 0) and fake points (label 1) in different colors
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