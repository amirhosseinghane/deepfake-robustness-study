# Robustness and Generalization Study for Deepfake Detection

A limited-scale empirical study on the **robustness** and **generalization** of
visual representations for detecting AI-generated (fake) faces, using a
**frozen pretrained network** and a **linear probe** protocol.

This repository contains the full, reproducible implementation of the study.

---

## Overview

This project evaluates how well frozen **ResNet-18** features (pretrained on
ImageNet) can distinguish real from fake face images, and how robust this
ability is against three common image degradations:

- **JPEG compression**
- **Gaussian noise**
- **Resolution reduction**

Instead of building a competitive detector, the goal is a controlled,
reproducible measurement of *relative* performance drop under degradation.

---

## Key Findings

- The representation is fairly robust to **compression** and **resolution
  reduction**, but **fragile against noise**.
- Training the linear probe with **noise** (the hardest degradation) yields
  the best **cross-degradation** robustness.
- Training with a **mix of all degradations** produces the most balanced,
  overall-robust model.

---

## Method

| Component | Choice |
|-----------|--------|
| Feature extractor | Frozen ResNet-18 (ImageNet-pretrained) |
| Classifier | Logistic Regression (linear probe) |
| Dataset | DF40-derived face classification set (test split, 3212 images, balanced) |
| Split | 80/20 train/test, `seed = 42`, stratified |
| Metrics | AUC, F1, Precision, Recall, confusion matrix |

All experiments use a **fixed random seed (42)** for reproducibility.

---

## Repository Structure

### Pipeline
| File | Purpose |
|------|---------|
| `make_manifest.py` | Builds a CSV manifest of all images |
| `extract_features.py` | Extracts frozen ResNet-18 features and caches them |
| `extract_degraded.py` | Extracts features for degraded train/test sets |
| `baseline.py` | Baseline evaluation on clean data |
| `toy_pipeline.py` | Small sanity-check run of the full pipeline |

### Robustness experiments
| File | Purpose |
|------|---------|
| `robustness_jpeg.py` | JPEG compression robustness |
| `robustness_noise.py` | Gaussian noise robustness |
| `robustness_resolution.py` | Resolution reduction robustness |
| `transfer_matrix.py` | Cross-degradation transfer study |
| `augmented_model.py` | Mixed-degradation augmentation experiment |

### Analysis & figures
| File | Purpose |
|------|---------|
| `full_metrics.py` | Full metric table (AUC/F1/Precision/Recall) + multi-seed |
| `roc_curves.py` | ROC curves per degradation |
| `tsne_plot.py` | t-SNE visualization of feature space |
| `error_analysis.py` | Error analysis (hardest / most confident mistakes) |
| `confusion_per_degradation.py` | Confusion matrix per degradation |
| `statistical_test.py` | Confidence intervals + paired significance test |
| `plot_robustness.py`, `plot_transfer.py` | Figure generation |

---

## Requirements

```
Python 3.x
torch, torchvision
scikit-learn
numpy, pandas
matplotlib, scipy
pillow
```

Install with:

```bash
pip install -r requirements.txt
```

---

## How to Run

The scripts expect the dataset to be placed in a `Data/test/` folder
(with `real/` and `fake/` subfolders) next to the code, then run in order:

```bash
python make_manifest.py       # build manifest.csv
python extract_features.py    # cache clean features -> features.npz
python extract_degraded.py    # cache degraded features -> degraded_features.npz
python baseline.py            # baseline metrics
python full_metrics.py        # full metric table + multi-seed
python roc_curves.py          # ROC figure
python tsne_plot.py           # t-SNE figure
python transfer_matrix.py     # transfer matrix
python augmented_model.py     # augmentation experiment
```

---

## Dataset

The dataset is a DF40-derived face classification set, publicly available on
Hugging Face. Only the balanced test split (1606 real + 1606 fake = 3212
images) is used, split internally 80/20 for train/test.

---

## Note

This is a limited-scale empirical study. Absolute accuracy is not the goal;
the focus is on *relative* robustness and generalization. See the accompanying
report for full analysis and limitations.
