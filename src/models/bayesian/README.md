
# Bayesian Naive Bayes Methods on PneumoniaMNIST

  

This repository implements and compares three variants of the Naive Bayes classifier on the [PneumoniaMNIST](https://medmnist.com/) dataset. In our Bayesian methodology, we distinguish between **Optimization Methods** and the **Naive Method**:

  

**Optimization Methods**

-  	**Gaussian Naive Bayes:** Uses normalized continuous features (pixel intensities scaled to [0, 1]) and assumes that these features follow a Gaussian distribution.

-  **Multinomial Naive Bayes:** Uses count features (original pixel values in the range 0–255) and models the data as discrete counts.

  

**Naive Method**

-  **Bernoulli Naive Bayes:** Uses binarized features (thresholded normalized pixel values) to model each pixel as either on or off. This is considered the classic or "naive" approach.

  

## Overview

  

This project demonstrates how different assumptions about input features affect classification performance on medical image data:

-  **Gaussian NB** and **Multinomial NB** (Optimization Methods) are designed for continuous or count data.

-  **Bernoulli NB** (Naive Method) is tailored for binary data.

  

## Dataset

  

[PneumoniaMNIST](https://medmnist.com/) is part of the MedMNIST collection and includes:

-  **Training Set:** 4708 images (28×28 pixels)

-  **Test Set:** 624 images (28×28 pixels)

-  **Classes:** "Normal" and "Pneumonia"

  

## Requirements

  

- Python 3.7+

- NumPy

- MedMNIST

- scikit-learn

- Matplotlib

  

Install the required packages with:

  

```bash

pip  install  numpy  medmnist  scikit-learn  matplotlib

```
## Methodology

 **Data Loading**
	The dataset is loaded using the MedMNIST API and the metadata from INFO. Images are of size 28×28 and labels indicate the presence or absence of pneumonia.

  

**Preprocessing**

  

 - **Gaussian NB:** Flatten images and normalize pixel values to [0, 1].

 - **Multinomial NB:** Flatten images and use the original integer pixel values (0–255).

 - **Bernoulli NB:** Use the normalized data and binarize it (e.g., threshold at 0.5).

3. Model Training and Evaluation:

	Each classifier is trained using its respective data format. Evaluation metrics include:
- Accuracy: Overall percentage of correctly classified samples.

- Confusion Matrix: Breakdown of true/false positives and negatives.

- Classification Report: Precision, recall, and F1-score for both classes.

## How to Run

1. Ensure you have installed the required packages.

2. Run the implementation script:
```
python naive_bayes_models.py
```
The script will:
- Download the PneumoniaMNIST dataset (if not already available).

- Preprocess the data for each Naive Bayes variant.

- Train the three classifiers.

- Print performance metrics and display confusion matrices.

## Evaluation Metrics

The models are evaluated using:

  

- Accuracy: Overall classification accuracy.

- Confusion Matrix: Detailed error analysis.

- Classification Report: Includes precision, recall, and F1-score for both "Normal" and "Pneumonia" classes.

These metrics help assess the performance of each model under its respective assumptions, which is critical for applications like medical diagnosis.


## References

- MedMNIST – A benchmark for medical image analysis.

- Murphy, K. P. Machine Learning: A Probabilistic Perspective – For an in-depth treatment of Bayesian decision theory.

- Scikit-learn Naive Bayes Documentation.