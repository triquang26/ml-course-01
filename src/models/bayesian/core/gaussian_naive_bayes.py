import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class MyGaussianNaiveBayes:
    """
    A simple implementation of Gaussian Naive Bayes .
    """
    def __init__(self, eps=1e-6):
        """
        Args:
            eps (float): Small value added to variances to avoid division by zero.
        """
        self.eps = eps
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.vars_ = None

    def fit(self, X, y):
        """
        Fits the Gaussian Naive Bayes model to the training data.
        
        Args:
            X (numpy.ndarray): Training features of shape (num_samples, num_features).
            y (numpy.ndarray): Training labels of shape (num_samples,).
        """
        # Identify unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Allocate space for priors, means, and variances
        self.priors_ = np.zeros(n_classes, dtype=np.float64)
        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.vars_ = np.zeros((n_classes, n_features), dtype=np.float64)

        # Compute class-specific statistics
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.priors_[idx] = X_c.shape[0] / float(X.shape[0])    # P(c)
            self.means_[idx, :] = X_c.mean(axis=0)
            # Variance + epsilon for numerical stability
            self.vars_[idx, :] = X_c.var(axis=0) + self.eps

    def predict(self, X):
        """
        Predicts class labels for samples in X.
        
        Args:
            X (numpy.ndarray): Test features of shape (num_samples, num_features).
        
        Returns:
            y_pred (numpy.ndarray): Predicted labels of shape (num_samples,).
        """
        # Compute predictions for each sample
        y_pred = [self._predict_single(sample) for sample in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        """
        Computes the predicted class for a single sample x.
        
        Args:
            x (numpy.ndarray): Feature vector of shape (num_features,).
        
        Returns:
            c (int or float): Predicted class label.
        """
        # Calculate log-posterior for each class
        posteriors = []
        for idx, c in enumerate(self.classes_):
            prior_log = np.log(self.priors_[idx])
            mean = self.means_[idx]
            var = self.vars_[idx]

            # Gaussian log-likelihood:
            # sum over features of [ -0.5 * log(2*pi*var_j) - ((x_j - mean_j)^2 / (2*var_j)) ]
            log_likelihood = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            log_likelihood -= 0.5 * np.sum(((x - mean) ** 2) / var)

            posterior = prior_log + log_likelihood
            posteriors.append(posterior)

        # Return class with the highest posterior
        return self.classes_[np.argmax(posteriors)]

def preprocess_data(train_dataset, test_dataset):
    """
    Preprocess data for Gaussian Naive Bayes.
    Flattens images and normalizes pixel values to [0,1].
    """
    # Extract images and labels
    X_train = train_dataset.imgs
    y_train = train_dataset.labels
    X_test = test_dataset.imgs
    y_test = test_dataset.labels
    
    # Flatten images and convert to float
    X_train_processed = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_test_processed = X_test.reshape(X_test.shape[0], -1).astype(np.float32)
    
    # Normalize pixel values to [0,1]
    X_train_processed /= 255.0
    X_test_processed /= 255.0
    
    # Ensure labels are 1D arrays
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    
    return X_train_processed, y_train, X_test_processed, y_test

def run_gaussian_naive_bayes(train_dataset, test_dataset):
    """
    Train and evaluate a Gaussian Naive Bayes model .
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        
    Returns:
        accuracy: Accuracy score on test set
        y_pred: Predictions on test set
        y_test: True labels for test set
        model: The trained Gaussian Naive Bayes model
    """
    # Preprocess data
    X_train, y_train, X_test, y_test = preprocess_data(train_dataset, test_dataset)
    
    print("\n" + "="*80)
    print("Running Gaussian Naive Bayes model ...")
    print("="*80)
    
    # Instantiate and train the model
    model = MyGaussianNaiveBayes(eps=1e-6)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=['Normal', 'Pneumonia'])
    
    print(f"Gaussian NB Results on Test Data ():")
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
        predictions: Model predictions
        actual_labels: True labels
        save_path: Path to save the visualization (optional)
    """
    cm = confusion_matrix(actual_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Normal", "Pneumonia"], 
                yticklabels=["Normal", "Pneumonia"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Gaussian Naive Bayes - Confusion Matrix")
    
    # Save the visualization if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()
