# Bayesian Naive Bayes Methods on PneumoniaMNIST

This repository implements and compares three variants of the Naive Bayes classifier on the [PneumoniaMNIST](https://medmnist.com/) dataset. In our Bayesian methodology, we distinguish between **Optimization Methods** and the **Naive Method**:

**Optimization Methods**
- **Gaussian Naive Bayes:** Uses normalized continuous features (pixel intensities scaled to [0, 1]) and assumes that these features follow a Gaussian distribution.
- **Multinomial Naive Bayes:** Uses count features (original pixel values in the range 0–255) and models the data as discrete counts.

**Naive Method**
- **Bernoulli Naive Bayes:** Uses binarized features (thresholded normalized pixel values) to model each pixel as either on or off. This is considered the classic or "naive" approach.

---

## Overview

This project demonstrates how different assumptions about input features affect classification performance on medical image data:
- **Gaussian NB** and **Multinomial NB** (Optimization Methods) are designed for continuous or count data.
- **Bernoulli NB** (Naive Method) is tailored for binary data.

---

## Dataset

[PneumoniaMNIST](https://medmnist.com/) is part of the MedMNIST collection and includes:
- **Training Set:** 4708 images (28×28 pixels)
- **Test Set:** 624 images (28×28 pixels)
- **Classes:** "Normal" and "Pneumonia"

The dataset is based on a prior collection of 5,856 pediatric chest X-Ray images. The task is binary classification of pneumonia versus normal. The source training set is split with a ratio of 9:1 into training and validation sets, and the source validation set is used as the test set. The images are originally grayscale, center-cropped, and resized into 1×28×28.

---

## Requirements

- Python 3.7+
- NumPy
- MedMNIST
- scikit-learn
- Matplotlib

Install the required packages with:

```bash
pip install numpy medmnist scikit-learn matplotlib
```


## Performance Analysis
**Metrics Implementation**
- Each model outputs accuracy, confusion matrices, and classification reports. These metrics help identify overall performance as well as class-specific performance (precision, recall, F1-score).

**Results Visualization**
- Confusion matrices are visualized with Seaborn heatmaps, offering an intuitive understanding of misclassifications and overall model behavior.

**Error Analysis**
- Detailed classification reports allow investigation of error types and potential biases in model predictions. This is especially important in a medical diagnosis context.

**Performance Comparison**
- A standardized evaluation pipeline enables direct comparisons among Gaussian, Bernoulli, and Multinomial variants. For example, our results show:

- Gaussian NB: Accuracy = 83.33%
- Bernoulli NB: Accuracy = 80.13%
- Multinomial NB: Accuracy = 83.33%

## Code Efficiency

**Time Complexity**
- The implementations utilize vectorized NumPy operations for most computations. The prediction loops can be further optimized with batch processing for large-scale data.

**Space Complexity**
- Data is stored as NumPy arrays; for PneumoniaMNIST, the memory footprint is modest. For larger datasets, additional optimizations or distributed processing might be needed.

**Resource Usage**
- The code is designed to run efficiently on standard CPU-based systems, requiring no specialized hardware.

**Optimization Attempts**
- Numerical stability is ensured through techniques like Laplace smoothing and the addition of small epsilon values. There is potential for further optimization, such as parallelizing the prediction loops.

**Scalability**
- While the current implementations are well-suited for the given dataset size, additional modifications (e.g., mini-batch processing) could facilitate scalability to larger datasets.

## Model Analysis

**Feature Importance**
- Although classical Naive Bayes models do not explicitly offer feature importance metrics, the computed per-class statistics (means, variances, and feature probabilities) offer insights into how individual pixels affect classification.

**Model Behavior Analysis**

By comparing confusion matrices and classification reports, one can observe how each model's assumptions impact performance:

- Gaussian NB: Handles continuous data robustly.
- Bernoulli NB: May lose subtle intensity variations due to binarization.
- Multinomial NB: Retains count-based information, leading to performance comparable to Gaussian NB.

**Limitation Discussion**
- The main limitation is the feature independence assumption, which may not fully hold for correlated image pixels. Additionally, preprocessing decisions (such as normalization and binarization) significantly affect model outcomes.

**Comparison with Theory**
- The implementations strictly follow theoretical principles, including log-space computations and Laplace smoothing, ensuring that the models are both interpretable and mathematically sound.

**Use Case Fit Analysis** 
- These models serve as effective baselines for medical image classification tasks. Their simplicity and transparency make them valuable for educational purposes and for understanding the basics of probabilistic classifiers in a medical context.