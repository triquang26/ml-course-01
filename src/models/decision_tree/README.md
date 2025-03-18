### Performance analysis
## Metrics implementation
The decision tree model is evaluated using the standard classification metrics
# Accuracy Score
Measures the percentage of correctly predicted labels over the total test samples.
Accuracy = Correct Predictions / Total samples
# Classification Report
Precision : True Positive / (True Positive + False Positive)
Recall : True Positive / (True Positive + False Negative)
F1-score : Harmonic mean of precision and recall
# Confusion Matrix
A matrix representation that shows how many instances were classified correctly and where the misclassifications occurred.

## Results visualization
# F1-Score Visualization
The F1-score heatmap is used to assess the classification performance per class.
The function plot_`f1_heatmap(y_true, y_pred)` plots a heatmap using `seaborn.heatmap()`.
The heatmap shows the F1-score for each class, highlighting which classes the model performs well on and which need improvement.
# Confusion Matrix Analysis
The confusion matrix heatmap provides a visual representation of the model's predictions.
The function `plot_confusion_matrix(y_true, y_pred)` uses `seaborn.heatmap()` to display counts of true vs. predicted labels.
This helps identify misclassification trends, such as:
False Positives (FP): Predicting pneumonia when it’s normal.
False Negatives (FN): Predicting normal when it’s pneumonia (a critical mistake).

## Error analysis
High FN: The model misses pneumonia cases, leading to untreated conditions.
Possible Fix: Adjust decision thresholds, increase false negative penalty.

High FP: Unnecessary medical interventions.
Possible Fix: Use class weighting or better feature selection.

Imbalance Issues: If "Normal" cases dominate, the model may favor them.
Possible Fix: Resampling or using F1-score instead of accuracy for evaluation.

## Performance comparision


# Model	                Accuracy	        Computational Cost	    Robustness	    Generalization      Ability	Interpretability
Decision Tree	        Moderate	        Low	                    Moderate	    Low-Moderate	    High
Graphical Model	        Moderate-High	    High	                High	        High	            Moderate
Neural Network	        Very High	        Very High	            High	        High	            Low
Genetic Algorithm	    Moderate-High	    Very High	            Moderate	    Moderate	        Low
Bayesian Model	        Moderate-High	    Low-Moderate	        High	        High	            Moderate


### Code efficiency
## Time complexity analysis
# Data Preprocessing
Mean and Std Computation (compute_mean_std)
Time Complexity:`O(N)`
Reason: The dataset is iterated once, and for each image, mean and standard deviation are computed.

Flattening Images `(X_train = np.array([img.numpy().flatten() for img, _ in train_dataset]))`
Time Complexity: `O(N⋅D)`
Reason: Each image (of size 𝐷) is flattened, requiring O(D) operations per image.

Train-Test Split (train_test_split)
Time Complexity: `O(N)`
Reason: The function randomly shuffles and splits the dataset once.
# Decision Tree Training 
Time Complexity:
Best Case (Balanced Tree): `O(NlogN)`
Worst Case (Unbalanced Tree): `O(N^2)`
Reason: Training a Decision Tree involves sorting and recursively splitting the dataset.
# Inference & Prediction (dt_model.predict)
Time Complexity
Best Case: `O(logN)` (Balanced tree)
Worst Case: `O(N)` (Unbalanced tree)

Reason: Prediction involves traversing the tree, which has a depth of at most O(logN) for a balanced tree.
# Evaluation Metrics Computation
F1-Score
Time Complexity: `O(N)`
Reason: The function iterates through all predictions to compute precision and recall.

Confusion Matrix 
Time Complexity: `O(N)`
Reason: Each prediction is compared to the ground truth label.

## Space complexity
# Data Storage
Original Image Data (train_dataset, test_dataset)
Space Complexity: O(ND)
Reason: Stores N images, each with D pixels.

Flattened Feature Matrix (X_train, X_test)
Space Complexity: `O(ND)`
Reason: Each image is stored as a 1D array of size D.

Labels 
Space Complexity: `O(N)`
Reason: One integer per sample.
# Decision Tree Model
Space Complexity:
Best Case (Balanced Tree): O(N)
Worst Case (Unbalanced Tree): O(N)
Reason: The tree stores N samples and corresponding splits.
# Confusion Matrix and F1-Score
Space Complexity: `O(C^2)` where `C` is the number of classes
Reason: The confusion matrix is a `C×C` matrix, and F1-score stores one value per class.

## Resource Usage
# CPU Usage
High usage during training (splitting nodes, sorting features).
Low usage during inference (tree traversal).
# Memory Usage
Moderate usage: Storing the dataset in memory can be expensive, especially with high-resolution images.
# Disk I/O
Loading the dataset from MedMNIST requires downloading and storing images on disk.


### Documentation
## API Documentation
# compute_mean_std(dataset)
Description: Computes the mean and standard deviation of the dataset for normalization.

Parameters:
dataset (torch.utils.data.Dataset): The dataset containing images.

Returns:
mean (float): Mean pixel intensity.
std (float): Standard deviation of pixel intensity.
# DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=42')
Description: Trains a Decision Tree classifier on the dataset.

Parameters:
criterion (str): "gini" or "entropy", specifies the split criterion.
max_depth (int): The maximum depth of the tree (to prevent overfitting).
random_state (int): Ensures reproducibility.

Methods:
`fit(X_train, y_train)`: Trains the model.
`predict(X_test`): Makes predictions on new data.
# f1_score(y_true, y_pred, average=None)
Description: Computes the F1-score for each class.

Parameters:
`y_true` (array): True labels.
`y_pred` (array): Predicted labels.
average (str): "None" returns F1-score for each class separately.

Returns:
numpy array: F1-score per class.
# confusion_matrix(y_true, y_pred)
Description: Computes the confusion matrix for model evaluation.

Parameters:
`y_true` (array): True labels.
`y_pred` (array): Predicted labels.

Returns:
2D numpy array: Confusion matrix.


### Model analysis
## Model behavior analysis
# Decision Boundaries in a Decision Tree
Decision trees create rectangular decision boundaries by splitting features at thresholds.
This works well for structured tabular data but struggles with image-based classification, where features are spatially related.
# Observing Overfitting
If max_depth is too high (e.g., 50), the tree memorizes training data → Overfitting.
If max_depth is too low (e.g., 3), the tree is too simple → Underfitting.
My chosen max_depth=10 is a tradeoff, but a random forest or CNN would generalize better.

## Limitations
Overfitting: Decision trees can memorize training data if depth is too high.
Lack of spatial awareness: Individual pixel intensity is used as a feature, but images have spatial structure that decision trees cannot fully capture.
Limited Generalization: Works well on small datasets, but struggles with large, diverse datasets.
Not Rotation or Scale Invariant: A CNN can recognize pneumonia even if an image is rotated or resized, but a decision tree cannot.
Computationally Inefficient for Large Data: A deep tree requires many comparisons per prediction, making it slower than other models on large datasets.

## Overcoming Limitations
Use Random Forests: Reduces overfitting by averaging multiple decision trees.





