# -*- coding: utf-8 -*-
# مرحله ۳: نسخه‌ی اسباب‌بازی
# هدف: اثبات اینکه کل زنجیره کار می‌کند، نه گرفتن دقت بالا
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- بذر تصادفی ثابت (برای بازتولیدپذیری) ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- مسیر مانیفست ---
MANIFEST = os.path.join(BASE_DIR, "manifest.csv")

# --- چند تصویر از هر کلاس برای این تست ---
N_PER_CLASS = 30

# ============================================================
# گام ۱: خواندن مانیفست و برداشتن نمونه‌ی کوچک
# ============================================================
df = pd.read_csv(MANIFEST)

# از هر کلاس N_PER_CLASS تصویر تصادفی برمی‌داریم
sample = (df.groupby("class_name", group_keys=False)
            .sample(n=N_PER_CLASS, random_state=SEED)
            .reset_index(drop=True))

print(f"Sample size: {len(sample)} images "
      f"({N_PER_CLASS} per class)")

# ============================================================
# گام ۲: آماده‌سازی مدل منجمد (ResNet-18)
# ============================================================
# مدل از پیش‌آموزش‌دیده روی ImageNet را بارگذاری می‌کنیم
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# لایه‌ی آخر (طبقه‌بند) را با یک لایه‌ی بی‌اثر جایگزین می‌کنیم
# تا خروجی مدل، بردار ویژگی باشد نه پیش‌بینی کلاس ImageNet
model.fc = nn.Identity()

# حالت ارزیابی: مدل را منجمد می‌کند (چیزی یاد نمی‌گیرد)
model.eval()

# --- تبدیل استاندارد تصویر برای ResNet ---
# تغییر اندازه به ۲۲۴، تبدیل به تنسور، و نرمال‌سازی ImageNet
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ============================================================
# گام ۳: استخراج ویژگی برای هر تصویر
# ============================================================
features = []  # بردارهای ویژگی
labels = []    # برچسب‌ها

# no_grad یعنی محاسبه‌ی گرادیان نکن (سریع‌تر و کم‌حافظه‌تر)
with torch.no_grad():
    for _, row in sample.iterrows():
        # باز کردن تصویر و تبدیل به RGB (بعضی تصاویر ممکن است حالت دیگری داشته باشند)
        img = Image.open(row["filepath"]).convert("RGB")

        # پیش‌پردازش و افزودن بُعد دسته (batch)
        tensor = preprocess(img).unsqueeze(0)

        # عبور از مدل و گرفتن بردار ویژگی
        feat = model(tensor).squeeze(0).numpy()

        features.append(feat)
        labels.append(row["label"])

features = np.array(features)
labels = np.array(labels)

print(f"Feature matrix shape: {features.shape}")
# انتظار: (60, 512) یعنی ۶۰ تصویر، هر کدام ۵۱۲ ویژگی

# ============================================================
# گام ۴: آموزش طبقه‌بند خطی و گزارش دقت
# ============================================================
clf = LogisticRegression(max_iter=1000)
clf.fit(features, labels)

# پیش‌بینی روی همان داده (فقط برای تست اینکه زنجیره کار می‌کند)
predictions = clf.predict(features)
acc = accuracy_score(labels, predictions)

print(f"Training accuracy (toy test): {acc:.3f}")
print("\nPipeline works end-to-end!")