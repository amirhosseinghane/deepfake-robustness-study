# -*- coding: utf-8 -*-
# Step 3: toy version (small test)
# goal: check that the whole pipeline works, not to get high accuracy
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

# --- fixed random seed (for reproducibility) ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- path to the manifest ---
MANIFEST = os.path.join(BASE_DIR, "manifest.csv")

# --- a few images per class for this test ---
N_PER_CLASS = 30

# ============================================================
# step 1: read the manifest and take a small sample
# ============================================================
df = pd.read_csv(MANIFEST)

# take N_PER_CLASS random images from each class
sample = (df.groupby("class_name", group_keys=False)
            .sample(n=N_PER_CLASS, random_state=SEED)
            .reset_index(drop=True))

print(f"Sample size: {len(sample)} images "
      f"({N_PER_CLASS} per class)")

# ============================================================
# step 2: set up the frozen model (ResNet-18)
# ============================================================
# load the ImageNet-pretrained model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# replace the last layer (classifier) with an identity layer
# so the output is a feature vector, not an ImageNet class
model.fc = nn.Identity()

# eval mode: freezes the model (no learning)
model.eval()

# --- standard image transform for ResNet ---
# resize to 224, convert to tensor, and ImageNet normalization
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ============================================================
# step 3: extract features for each image
# ============================================================
features = []  # feature vectors
labels = []    # labels

# no_grad means no gradient computation (faster, less memory)
with torch.no_grad():
    for _, row in sample.iterrows():
        # open the image and convert to RGB (some images may be in another mode)
        img = Image.open(row["filepath"]).convert("RGB")

        # preprocess and add a batch dimension
        tensor = preprocess(img).unsqueeze(0)

        # pass through the model and get the feature vector
        feat = model(tensor).squeeze(0).numpy()

        features.append(feat)
        labels.append(row["label"])

features = np.array(features)
labels = np.array(labels)

print(f"Feature matrix shape: {features.shape}")
# expected: (60, 512) means 60 images, each with 512 features

# ============================================================
# step 4: train the linear classifier and report accuracy
# ============================================================
clf = LogisticRegression(max_iter=1000)
clf.fit(features, labels)

# predict on the same data (just to test the pipeline works)
predictions = clf.predict(features)
acc = accuracy_score(labels, predictions)

print(f"Training accuracy (toy test): {acc:.3f}")
print("\nPipeline works end-to-end!")