import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

DEGRADED = os.path.join(BASE_DIR, "degraded_features.npz")

deg = np.load(DEGRADED)
y_train = deg["y_train"]
y_test = deg["y_test"]

# Train classifier on clean training features
clf = LogisticRegression(max_iter=1000)
clf.fit(deg["Xtrain_clean"], y_train)

conditions = {
    "Clean": "Xtest_clean",
    "JPEG (Q=20)": "Xtest_jpeg",
    "Noise (sigma=30)": "Xtest_noise",
    "Resolution (x0.35)": "Xtest_resolution",
}

print("Confusion matrices per degradation")
print("Rows = True, Columns = Predicted")
print("Order: [Real, Fake]\n")

for name, key in conditions.items():
    X_te = deg[key]
    y_pred = clf.predict(X_te)
    cm = confusion_matrix(y_test, y_pred)
    print(f"--- {name} ---")
    print(f"              Pred_Real  Pred_Fake")
    print(f"  True_Real   {cm[0,0]:>8}   {cm[0,1]:>8}")
    print(f"  True_Fake   {cm[1,0]:>8}   {cm[1,1]:>8}")
    print()

print("Done.")