# check which images the model gets wrong most
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

# load features and manifest (for the file paths)
data = np.load(FEATURES)
X, y = data["features"], data["labels"]
df = pd.read_csv(MANIFEST)

# the usual split
indices = np.arange(len(X))
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X, y, indices, test_size=0.2, random_state=SEED, stratify=y)

# train and predict
clf = LogisticRegression(max_iter=1000)
clf.fit(X_tr, y_tr)
y_prob = clf.predict_proba(X_te)[:, 1]
y_pred = clf.predict(X_te)

# ============================================================
# step 1: find the errors
# ============================================================
errors = (y_pred != y_te)
n_errors = errors.sum()
print(f"Total test samples: {len(y_te)}")
print(f"Total errors: {n_errors} ({100*n_errors/len(y_te):.1f}%)")

# separate the two error types
false_positives = (y_pred == 1) & (y_te == 0)  # real predicted as fake
false_negatives = (y_pred == 0) & (y_te == 1)  # fake predicted as real
print(f"  False Positives (real called fake): {false_positives.sum()}")
print(f"  False Negatives (fake called real): {false_negatives.sum()}")

# ============================================================
# step 2: analyze the model confidence on errors
# ============================================================
# distance of probability from 0.5 = confidence
confidence = np.abs(y_prob - 0.5)

error_conf = confidence[errors].mean()
correct_conf = confidence[~errors].mean()
print(f"\nMean confidence on correct predictions: {correct_conf:.3f}")
print(f"Mean confidence on errors:              {error_conf:.3f}")
print("(lower confidence on errors means the model was unsure when wrong)")

# ============================================================
# step 3: hardest samples (most confident mistakes)
# ============================================================
# errors where the model was most confident but wrong
error_indices = np.where(errors)[0]
error_confidences = confidence[error_indices]
# sort by confidence (descending)
hardest = error_indices[np.argsort(-error_confidences)][:10]

print(f"\n--- 10 hardest errors (most confident mistakes) ---")
print(f"{'File':>40} | {'True':>5} | {'Pred':>5} | {'Prob':>6}")
print("-"*65)
for i in hardest:
    global_idx = idx_te[i]
    filepath = df.iloc[global_idx]["filepath"]
    filename = filepath.split("\\")[-1][:38]  # file name only
    true_label = "fake" if y_te[i] == 1 else "real"
    pred_label = "fake" if y_pred[i] == 1 else "real"
    print(f"{filename:>40} | {true_label:>5} | {pred_label:>5} | {y_prob[i]:>6.3f}")

# ============================================================
# step 4: save a summary for the report
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
