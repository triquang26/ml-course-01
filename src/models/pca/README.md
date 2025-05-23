# PCA Evaluation Utilities
This repository provides utility functions to train and evaluate a Support Vector Machine (SVM) classifier with Principal Component Analysis (PCA) for dimensionality reduction on the PneumoniaMNIST dataset. The primary tools include:

* **Confusion Matrix**: Computes and visualizes the confusion matrix for true vs. predicted labels (Normal vs. Pneumonia).
* **PCA Explained Variance Plot**: Visualizes the cumulative explained variance ratio of PCA components to assess information retention.

## Prerequisites
* Python 3.7 or higher
* numpy 1.18+
* matplotlib 3.x
* seaborn 0.10+
* scikit-learn 0.22+
* joblib for model persistence
* medmnist for PneumoniaMNIST dataset

Background
This pipeline combines PCA and SVM for classifying PneumoniaMNIST images:

PCA Dimensionality Reduction: Reduces 784-dimensional vectors (28x28 flattened images) to a smaller set of principal components (default: 50), retaining most data variance.
SVM Classification: Uses an SVM with an RBF kernel to classify PCA-transformed data into Normal or Pneumonia categories.
Evaluation: Computes accuracy, F1-score, macro F1-score, and visualizes performance via confusion matrices and PCA variance plots.
In this implementation:

Data Preparation: Each 28x28 image is flattened into a 784-dimensional vector, standardized, and reduced using PCA to a lower-dimensional space.
Classification: An SVM model is trained on PCA-transformed training data and predicts labels for test data based on the learned decision boundary.
Evaluation: Confusion matrices show classification performance, while PCA variance plots indicate how much information is retained after dimensionality reduction.
This approach leverages PCA to improve computational efficiency and SVM to perform robust classification.

# Implementation Details
The evaluation and visualization functions rely on a shared pipeline:

Data Preprocessing: Images are flattened, standardized with StandardScaler, and reduced to n_components via PCA (load_data and PneumoniaPredictor.preprocess_data).
Model Training: The SVM model is trained on PCA-transformed data using an RBF kernel (run_svm_model).
Prediction: The PneumoniaPredictor class preprocesses new data with PCA and predicts labels using the trained SVM (predict, predict_single).
Evaluation: Metrics (accuracy, F1-score) are computed, and visualizations (confusion matrix, PCA variance plot) are generated using visualize_results and visualize_pca_variance.
This design ensures modularity and extensibility for preprocessing, training, and evaluation.

# Result
Below is the classification report on the test set, based on the run_svm_model evaluation:
Classification Report on Test Set:
              precision    recall  f1-score   support
      Normal       0.63      0.99      0.77       150
   Pneumonia      0.99      0.82      0.90       474
    accuracy                           0.86       624
   macro avg       0.81      0.90      0.83       624
weighted avg       0.91      0.86      0.87       624

# Note-result
"Metric"              "Value"
"Normal F1-score"      0.77
"Pneumonia F1-score"   0.90
"Overall accuracy"     0.86
"Total samples"        624

