# Step 5: baseline
# load the saved features, split into train/test,
# train the classifier and report the real performance
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix

# --- fixed random seed ---
SEED = 42
np.random.seed(SEED)

# --- path to the features file ---
FEATURES = os.path.join(BASE_DIR, "features.npz")

# ============================================================
# step 1: load features and labels
# ============================================================
data = np.load(FEATURES)
X = data["features"]   # feature vectors
y = data["labels"]     # labels (0=real, 1=fake)

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# ============================================================
# step 2: split into train and test
# ============================================================
# 20% of the data is kept for testing.
# stratify=y keeps the class ratio in both splits.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

print(f"\nTrain size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# ============================================================
# step 3: train the linear classifier
# ============================================================
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# ============================================================
# step 4: evaluate on the test data (unseen by the model)
# ============================================================
# predict the class
y_pred = clf.predict(X_test)

# probability of the positive class (for AUC)
y_prob = clf.predict_proba(X_test)[:, 1]

# --- metrics ---
auc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n--- Baseline Results (on unseen test data) ---")
print(f"AUC:      {auc:.4f}")
print(f"F1:       {f1:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"\nConfusion Matrix:")
print(f"              Pred_Real  Pred_Fake")
print(f"  True_Real   {cm[0,0]:>8}   {cm[0,1]:>8}")
print(f"  True_Fake   {cm[1,0]:>8}   {cm[1,1]:>8}")
