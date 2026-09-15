# -*- coding: utf-8 -*-
# اصلاحیه استاد: فاصله اطمینان و آزمون معناداری آماری
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy import stats

FEATURES = os.path.join(BASE_DIR, "features.npz")
DEGRADED = os.path.join(BASE_DIR, "degraded_features.npz")

# چند seed برای ساختن توزیع
SEEDS = [42, 7, 123, 2024, 99, 555, 888, 1000, 314, 271]

# ============================================================
# گام ۱: فاصله اطمینان برای خط مبنا (چند seed)
# ============================================================
data = np.load(FEATURES)
X, y = data["features"], data["labels"]

aucs = []
for seed in SEEDS:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    y_prob = clf.predict_proba(X_te)[:, 1]
    aucs.append(roc_auc_score(y_te, y_prob))

aucs = np.array(aucs)
mean_auc = aucs.mean()
std_auc = aucs.std()

# فاصله اطمینان ۹۵٪ (با فرض توزیع t)
ci = stats.t.interval(0.95, len(aucs)-1,
                      loc=mean_auc, scale=stats.sem(aucs))

print("="*55)
print(f"Baseline AUC over {len(SEEDS)} seeds")
print("="*55)
print(f"Mean AUC:        {mean_auc:.4f}")
print(f"Std:             {std_auc:.4f}")
print(f"95% CI:          [{ci[0]:.4f}, {ci[1]:.4f}]")

# ============================================================
# گام ۲: مقایسه‌ی آماری دو مدل (سالم در برابر augmented)
# ============================================================
# آیا تفاوت augmented و clean معنادار است؟
deg = np.load(DEGRADED)
y_train, y_test = deg["y_train"], deg["y_test"]
conditions = ["noise", "jpeg", "resolution"]

# مدل سالم: AUC روی هر تخریب
clf_clean = LogisticRegression(max_iter=1000)
clf_clean.fit(deg["Xtrain_clean"], y_train)
clean_aucs = [roc_auc_score(y_test,
              clf_clean.predict_proba(deg[f"Xtest_{c}"])[:, 1])
              for c in conditions]

# مدل augmented (ترکیب همه تخریب‌ها)
np.random.seed(42)
n_train = len(y_train)
choice = np.random.randint(0, 4, size=n_train)
all_conds = ["clean", "noise", "jpeg", "resolution"]
X_aug = np.zeros_like(deg["Xtrain_clean"])
for idx, c in enumerate(all_conds):
    mask = (choice == idx)
    X_aug[mask] = deg[f"Xtrain_{c}"][mask]
clf_aug = LogisticRegression(max_iter=1000)
clf_aug.fit(X_aug, y_train)
aug_aucs = [roc_auc_score(y_test,
            clf_aug.predict_proba(deg[f"Xtest_{c}"])[:, 1])
            for c in conditions]

print("\n" + "="*55)
print("Clean-trained vs Augmented model (on degraded tests)")
print("="*55)
print(f"{'Condition':>12} | {'Clean':>7} | {'Augmented':>9}")
print("-"*35)
for i, c in enumerate(conditions):
    print(f"{c:>12} | {clean_aucs[i]:>7.4f} | {aug_aucs[i]:>9.4f}")

# آزمون t زوجی
t_stat, p_value = stats.ttest_rel(aug_aucs, clean_aucs)
print(f"\nPaired t-test (Augmented vs Clean):")
print(f"  Mean difference: {np.mean(aug_aucs) - np.mean(clean_aucs):+.4f}")
print(f"  p-value: {p_value:.4f}")
if p_value < 0.05:
    print("  → تفاوت از نظر آماری معنادار است (p < 0.05)")
else:
    print("  → تفاوت از نظر آماری معنادار نیست (p ≥ 0.05)")
    print("  (به دلیل تعداد کم نقاط مقایسه، طبیعی است)")

print("\nDone.")