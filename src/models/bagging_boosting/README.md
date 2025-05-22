# Bagging & Boosting Methods on PneumoniaMNIST

This repository implements and compares three popular ensemble methods—**Bagging**, **AdaBoost**, and **Gradient Boosting**—on the [PneumoniaMNIST](https://medmnist.com/) dataset, each in two flavors:

1. **scikit-learn implementations**  
2. **From-scratch implementations** (in `src/models/bagging_boosting/core/`)

---

## Overview

Ensemble methods combine many “weak” learners (in our case, decision trees) to produce a stronger overall predictor. We cover:
| Model                | Method |
|----------------------|:-----------------------|
|**Bagging (Bootstrap Aggregating)**  | Trains each tree on a bootstrap sample and averages/votes.
|**AdaBoost.M1 (Adaptive Boosting)**  | Iteratively reweights training examples to focus on the hard cases.
|**Gradient Boosting**                | Fits each new tree to the residuals (negative gradients) of the previous ensemble.

We implement each algorithm both via **scikit-learn** and **from scratch**, then compare:

- **Base Decision Tree** (no ensemble)
- **SK Bagging**, **SK AdaBoost**, **SK GradientBoosting**
- **Scratch Bagging**, **Scratch AdaBoost**, **Scratch GradBoost**

---

## Dataset

We use [PneumoniaMNIST](https://medmnist.com/) from the MedMNIST collection:

- **Training set:** 4,708 chest X-ray images (28×28, grayscale)  
- **Test set:** 624 images  
- **Classes:** `normal` vs. `pneumonia`  

The `src/data/preprocess/bagging_boosting.py` module downloads and loads the train/test splits.

---

## Requirements

- Python 3.7+  
- NumPy  
- scikit-learn  
- Matplotlib  
- MedMNIST  

Install with:

```bash
pip install numpy scikit-learn matplotlib medmnist
```

## Performance Analysis

| Model                | Accuracy (scikit-learn) | Accuracy (Scratch) |
|----------------------|:-----------------------:|:------------------:|
| Base Decision Tree   |                  0.8173 |                n/a |
| SK Bagging           |                  0.8205 |             0.8269 |
| SK AdaBoost          |                  0.8478 |             0.8478 |
| SK GradientBoost     |                  0.8317 |             0.8205 |


- Best overall: SK and Scratch  AdaBoost at ~84.8 %.

- Scratch vs. SK: Bagging/AdaBoost scratch versions closely match; scratch Gradient Boosting is slightly lower due to simplified step‐size handling.

- Confusion matrices and full classification reports for each run are saved under reports/figures/.

## Code Efficiency
- scikit-learn’s C‐optimized trees, line searches, and learning‐rate tuning yield strong out‐of‐the‐box performance.

- Scratch versions rely on Python loops and fixed step sizes (γ = 1), which can underfit or over/under‐shoot.

- All methods vectorize data loading and scoring; the scratch learners can be accelerated by JIT or Cython.

## Model Analysis
- Bagging reduces variance via bootstrap aggregation—scratch and SK versions nearly match.

- AdaBoost focuses successive learners on misclassified points—both versions achieve ≈ 84 %.

- Gradient Boosting iteratively fits to residuals (negative gradients of binomial deviance). Our scratch version uses a fixed step size and no line search, which lowers accuracy relative to scikit-learn’s optimized implementation.

## Limitations & Next Steps
- Scratch GradientBoost:

    - Implement per-stage line search for optimal step size γₘ.

    - Support subsample, min_samples_leaf, and other regularization parameters.

    - Add early-stopping or validation monitoring.

- Enhance Preprocessing:

    - Histogram equalization, data augmentation (e.g. flips, rotations).

    - Explore feature extraction (e.g. simple CNN backbones).

- Hyperparameter Tuning:

    - Grid or randomized search over n_estimators, learning_rate, max_depth, and subsample.