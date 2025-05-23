# HMM Evaluation Utilities

This repository provides utility functions to evaluate a Hidden Markov Model (HMM) classifier built using `GaussianHMM`. The primary evaluation tools include:

* **F1-Score Heatmap**: Computes per-class F1 scores and visualizes them in a heatmap.
* **Confusion Matrix**: Computes and visualizes the confusion matrix for true vs. predicted labels.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Usage](#usage)

   * [F1-Score Heatmap](#f1-score-heatmap)
   * [Confusion Matrix](#confusion-matrix)
4. [Function Signatures](#function-signatures)
5. [License](#license)

---

## Prerequisites

* Python 3.7 or higher
* `numpy` 1.18+
* `matplotlib` 3.x
* `seaborn` 0.10+
* `scikit-learn` 0.22+
* `hmmlearn` for `GaussianHMM`

Install dependencies via pip:

```bash
pip install numpy matplotlib seaborn scikit-learn hmmlearn
```

---

## Usage

Import the evaluation utilities and call them with your trained HMM models and test data.

```python
from utils.hmm_evaluate import plot_hmm_f1_heatmap, plot_hmm_confusion_matrix

# `models` should be a dict mapping label -> trained GaussianHMM model
# `X_test` is an array of shape (N, D), where D is sequence length
# `y_test` is the true label array of length N

# 1) F1-Score Heatmap
plot_hmm_f1_heatmap(models=models, X=X_test, y=y_test, class_labels=['0', '1', ..., '9'])

# 2) Confusion Matrix
plot_hmm_confusion_matrix(models=models, X=X_test, y=y_test, class_labels=['0', '1', ..., '9'])
```

These functions will display the plots inline (e.g., in a Jupyter notebook) or in a pop-up window depending on your environment.

---

---

## Background

Hidden Markov Models (HMMs) are statistical sequence models characterized by:

1. **Latent States**: A discrete set of hidden states that evolve through a Markov chain with state transition probabilities.
2. **Emissions**: Observations generated from each hidden state according to a probability distribution – here, Gaussians over pixel intensities.
3. **Inference**: Given an observed sequence, the HMM computes the likelihood of that sequence for each model via the forward (or Viterbi) algorithm.

In this implementation:

* **Data Preparation**: Each MedMNist image (28×28 pixels) is flattened into a 784‑dimensional vector. We treat this as a single sequence of 784 timesteps, where each “timestep” emits one pixel intensity.
* **Per‑Class Modeling**: We train one `GaussianHMM` per class (two models for binary classification) on training images belonging to that class.
* **Classification**: At test time, each image sequence is scored against both HMMs; the model yielding the higher log‑likelihood assigns the predicted label.

This approach leverages the temporal‑sequence machinery of HMMs to model spatial structure in images by imposing an arbitrary but fixed scan order over pixels.

---

## Implementation Details

Both evaluation functions rely on the same internal prediction pipeline:

1. **Sequence Preparation**: Each feature vector `x` of shape `(D,)` is reshaped to `(D, 1)` so it can be treated as a time series sequence by the HMM.
2. **Scoring**: For each class model in the `models` dict, the log‑likelihood of the sequence is computed via the `GaussianHMM.score()` method.
3. **Prediction**: The class whose model yields the highest log‑likelihood is chosen as the predicted label for that sequence.
4. **Aggregation**: All sample predictions are collected into a 1D array, which is then passed to `sklearn.metrics.f1_score()` or `confusion_matrix()` for metric computation.

This design ensures that the evaluation functions remain lightweight wrappers around a shared, easily extensible prediction loop.

---

## Function Signatures

```python
def plot_hmm_f1_heatmap(
    models: Dict[Any, GaussianHMM],
    X: np.ndarray,
    y: np.ndarray,
    class_labels: Optional[List[str]] = None
) -> None:
    """
    Computes per-class F1 scores from an HMM classifier and plots them as a heatmap.

    Args:
      - models: Dict[label -> GaussianHMM] trained HMMs for each class.
      - X: N×D numpy array of flattened sequences (each x reshaped to D×1).
      - y: Array of true labels of length N.
      - class_labels: Optional list of class names for heatmap ticks.
    """
```

```python
def plot_hmm_confusion_matrix(
    models: Dict[Any, GaussianHMM],
    X: np.ndarray,
    y: np.ndarray,
    class_labels: Optional[List[str]] = None
) -> None:
    """
    Computes and plots the confusion matrix for an HMM classifier.

    Args:
      - models: Dict[label -> GaussianHMM] trained HMMs for each class.
      - X: N×D numpy array of flattened sequences (each x reshaped to D×1).
      - y: Array of true labels of length N.
      - class_labels: Optional list of class names for axes ticks.
    """
```

---

## Result

Below is the classification report on the test set:

```
              precision    recall  f1-score   support

      normal       0.72      0.66      0.69       234
   pneumonia       0.81      0.85      0.83       390

    accuracy                           0.78       624
   macro avg       0.77      0.76      0.76       624
weighted avg       0.78      0.78      0.78       624
```

You can also summarize key metrics:

| Metric                 | Value |
| ---------------------- | ----- |
| **Normal F1 score**    | 0.69  |
| **Pneumonia F1 score** | 0.83  |
| **Overall accuracy**   | 0.78  |
| **Total samples**      | 624   |
