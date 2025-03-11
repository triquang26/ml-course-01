# GA-Optimized Ensemble Learning for PneumoniaMNIST

This repository demonstrates an ensemble learning approach for binary classification on the PneumoniaMNIST dataset. The ensemble combines three different models—a Decision Tree, a Bayesian (Naive Bayes) model, and a Convolutional Neural Network (CNN)—with the ensemble weights optimized using a Genetic Algorithm (GA).

## Overview

The project uses the complementary strengths of different classifiers:
- **Decision Tree:** Provides an interpretable, non-linear decision boundary.
- **Bayesian Model (Naive Bayes):** Offers a probabilistic perspective that works well even with small datasets.
- **CNN:** Excels at extracting and learning features directly from image data.

The Genetic Algorithm optimizes the weights used to combine the prediction probabilities of these models, maximizing the overall classification performance on a validation set.

## How to run
**Install dependencies using:** `pip install -r requirements.txt` (Remember to use the same environment for the .ipynb file). And then just run the ipynb file

## Motivation
The primary motivation behind this approach is to harness the strengths of different models and improve overall predictive performance:

**Model Diversity:** The Decision Tree, Bayesian model, and CNN provide different perspectives on the data. While the CNN excels in feature extraction, the Decision Tree and Bayesian methods offer interpretability and robust probabilistic predictions.

**Automated Weight Tuning:** Manually tuning ensemble weights can be challenging. Using a Genetic Algorithm automates this process, effectively exploring the weight space to find an optimal combination.

**Robust Performance:** Ensemble methods can mitigate the weaknesses of individual models by averaging their predictions, which is particularly important in medical image classification tasks like detecting pneumonia.


## Advantages and Disadvantages:
### Advantages:
**Enhanced Predictive Performance:** Combining models often leads to better accuracy and robustness compared to any single model.
**Automated Optimization:** The Genetic Algorithm reduces the need for manual hyperparameter tuning by automatically finding optimal ensemble weights.
**Complementary Strengths:** The method leverages both interpretable models (Decision Tree, Bayesian) and a powerful feature extractor (CNN).
### Disadvantages:
**Computational Overhead:** Training multiple models and performing GA optimization increases computational cost and runtime.
**Complexity:** The ensemble framework introduces additional complexity in both model management and debugging.
**Risk of Overfitting:** Without proper regularization and cross-validation, the ensemble might overfit the validation data, as suggested by a slight discrepancy between validation and test performance.
**Validation Dependency:** The success of the GA is highly dependent on the chosen validation strategy and performance metric.

## Fitness Function and Overfitting Prevention
A key aspect of our approach is the design of the fitness function used in the Genetic Algorithm, which is crafted to prevent overfitting. The fitness function evaluates the performance of a candidate set of ensemble weights based on the average AUC across cross-validation folds. Importantly, it incorporates an L2 regularization term that penalizes extreme weight values:
**Regularization Term:** The term `reg_penalty = reg_lambda * np.sum(np.square(weights))` is subtracted from the average AUC. This discourages any single model from dominating the ensemble, promoting a balanced contribution from all classifiers.
**Prevention of Overfitting:** By penalizing overly large weights, the fitness function reduces the risk of overfitting to the validation set. This ensures that the GA not only finds a set of weights that perform well on the validation data but also generalize better to unseen test data.

This design choice helps maintain a balance between model complexity and generalization performance.

## Evaluation:
### Resutls Analysis:
The Genetic Algorithm optimization yielded the following key results:
* **Optimized Ensemble Weights**: The GA converged to weights approximately: `[-0.00666, 0.33158, 0.08182]`, with a validation accuracy of **98.67%**.
* **Test Set Performance**: The final ensemble achieved a test accuracy of **82.37%**.
* **F1 Scores**:
    * **Decision Tree**: 0.8575
    * **Bayesian Model**: 0.8667
    * **CNN**: 0.8661
    * **Ensemble: 0.8756**

* **Confustion Matrices:**
    * **Decision Tree:**
    ```
        [[131 103]
        [ 20 370]]
    ```

    * **Bayesian Model:**
    ```
        [[169  65]
        [ 42 348]]
    ```
    * **CNN:**
    ```
        [[116 118]
        [  2 388]]
    ```
    * **Ensemble:**
    ```
        [[127 107]
        [  3 387]]
    ```
### Discussion:
**Improved F1 Score:** The ensemble model achieved the highest F1 score (0.8756), indicating a better balance between precision and recall compared to the individual models.
**Minimized False Negatives:** With only 3 false negatives, the ensemble significantly reduces the risk of overlooking pneumonia cases—a critical factor in medical diagnosis.
**Trade-Offs:** Although the ensemble method improves performance, it introduces additional complexity and computational demands. The slight gap between high validation accuracy and lower test accuracy suggests that further tuning of GA parameters (like mutation rate or regularization strength) may be beneficial.


