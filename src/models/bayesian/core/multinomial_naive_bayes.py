import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class MyMultinomialNB:
    """
    A simple implementation of Multinomial Naive Bayes .
    This implementation assumes that the features are count data (non-negative integers).
    """
    def __init__(self, alpha=1.0):
        """
        Args:
            alpha (float): Laplace smoothing parameter.
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None  # Log prior probabilities for each class
        self.feature_log_prob_ = None  # Log probabilities P(feature | class)
    
    def fit(self, X, y):
        """
        Fit the multinomial Naive Bayes model according to the given training data.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features) with integer counts.
            y (np.ndarray): Training labels of shape (n_samples,).
        """
        # Identify unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Initialize count matrix for features per class and class counts
        class_count = np.zeros(n_classes, dtype=np.float64)
        feature_count = np.zeros((n_classes, n_features), dtype=np.float64)
        
        # Aggregate counts for each class
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            class_count[idx] = X_c.shape[0]
            feature_count[idx, :] = X_c.sum(axis=0)
        
        # Compute log prior: log(P(c)) for each class
        self.class_log_prior_ = np.log(class_count) - np.log(class_count.sum())
        
        # Compute smoothed feature probabilities
        smoothed_fc = feature_count + self.alpha
        smoothed_feature_sum = smoothed_fc.sum(axis=1).reshape(-1, 1)  # Sum per class
        # log P(feature|class)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_feature_sum)
    
    def predict(self, X):
        """
        Perform classification on an array of test vectors X.
        
        Args:
            X (np.ndarray): Test data of shape (n_samples, n_features).
        
        Returns:
            C (np.ndarray): Predicted target values for X.
        """
        # Compute the joint log likelihood of each sample for each class:
        jll = self._joint_log_likelihood(X)
        # Choose the class with highest likelihood
        return self.classes_[np.argmax(jll, axis=1)]
    
    def _joint_log_likelihood(self, X):
        """
        Calculate the posterior log probability of the samples X.
        
        Args:
            X (np.ndarray): Input data.
        
        Returns:
            jll (np.ndarray): Array of shape (n_samples, n_classes) with joint log likelihood.
        """
        # For each sample, jll = log(P(c)) + sum_j x_j * log(P(feature_j|c))
        return X @ self.feature_log_prob_.T + self.class_log_prior_

def preprocess_data(train_dataset, test_dataset):
    """
    Preprocess data for Multinomial Naive Bayes.
    Flattens images and keeps original integer pixel values (0 to 255).
    """
    # Extract images and labels
    X_train = train_dataset.imgs
    y_train = train_dataset.labels
    X_test = test_dataset.imgs
    y_test = test_dataset.labels
    
    # Flatten images and ensure integer format
    X_train_processed = X_train.reshape(X_train.shape[0], -1).astype(np.int32)
    X_test_processed = X_test.reshape(X_test.shape[0], -1).astype(np.int32)
    
    # Ensure labels are 1D arrays
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    
    return X_train_processed, y_train, X_test_processed, y_test

def run_multinomial_naive_bayes(train_dataset, test_dataset):
    """
    Train and evaluate a Multinomial Naive Bayes model ().
    
    Args:
        train_dataset: Training dataset.
        test_dataset: Test dataset.
        
    Returns:
        accuracy: Accuracy score on test set.
        y_pred: Predictions on test set.
        y_test: True labels for test set.
        model: The trained Multinomial Naive Bayes model.
    """
    # Preprocess data
    X_train, y_train, X_test, y_test = preprocess_data(train_dataset, test_dataset)
    
    print("\n" + "="*80)
    print("Running Multinomial Naive Bayes model ...")
    print("="*80)
    
    # Train the model
    model = MyMultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=['Normal', 'Pneumonia'])
    
    print(f"Multinomial NB Results on Test Data :")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    
    return accuracy, y_pred, y_test, model

def visualize_results(predictions, actual_labels, save_path=None):
    """
    Visualize results using confusion matrix.
    
    Args:
        predictions: Model predictions.
        actual_labels: True labels.
        save_path: Path to save the visualization.
    """
    cm = confusion_matrix(actual_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Normal", "Pneumonia"], 
                yticklabels=["Normal", "Pneumonia"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Multinomial Naive Bayes - Confusion Matrix")
    
    # Save the visualization if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()
