# Axis 2 - part 2: build the transfer matrix
# compute an AUC for each pair (train on X, test on Y)
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEED = 42
np.random.seed(SEED)

# --- path to the degraded features file ---
DEGRADED = os.path.join(BASE_DIR, "degraded_features.npz")
BASE = BASE_DIR

# ============================================================
# step 1: load all features
# ============================================================
data = np.load(DEGRADED)
y_train = data["y_train"]
y_test = data["y_test"]

# degradation conditions
conditions = ["clean", "noise", "jpeg", "resolution"]

# ============================================================
# step 2: compute AUC for each train x test pair
# ============================================================
# make an empty table: rows = train, cols = test
matrix = pd.DataFrame(index=conditions, columns=conditions, dtype=float)

for train_cond in conditions:
    # training features with this degradation
    X_train = data[f"Xtrain_{train_cond}"]

    # train the classifier once
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # now evaluate on all test conditions
    for test_cond in conditions:
        X_test = data[f"Xtest_{test_cond}"]
        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        matrix.loc[train_cond, test_cond] = auc

# ============================================================
# step 3: show and save the matrix
# ============================================================
print("\n=== Transfer Matrix (AUC) ===")
print("Rows = trained on | Columns = tested on\n")
print(matrix.round(4).to_string())

matrix.to_csv(BASE + r"\transfer_matrix.csv")
print("\nSaved: transfer_matrix.csv")

# ============================================================
# step 4: some automatic analysis to help interpretation
# ============================================================
print("\n--- Quick analysis ---")

# average performance when trained on clean (clean row)
print(f"Trained on CLEAN, avg on degraded tests: "
      f"{matrix.loc['clean', ['noise','jpeg','resolution']].mean():.4f}")

# average of each row over the degraded tests
for cond in conditions:
    avg_on_degraded = matrix.loc[cond, ['noise','jpeg','resolution']].mean()
    print(f"Trained on {cond:>10}: avg AUC on degraded tests = {avg_on_degraded:.4f}")
