import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class MyBernoulliNaiveBayes:
    """
    A simple implementation of Bernoulli Naive Bayes .
    Bernoulli NB assumes each feature is binary (0 or 1).
    """
    def __init__(self, alpha=1.0):
        """
        Args:
            alpha (float): Laplace smoothing parameter.
        """
        self.alpha = alpha
        self.classes_ = None
        self.priors_ = None
        self.feature_probs_ = None  # theta_{c,j} for each class c and feature j

    def fit(self, X, y):
        """
        Fits the Bernoulli Naive Bayes model to the training data.
        
        Args:
            X (numpy.ndarray): Binary training features of shape (num_samples, num_features).
                              Each entry should be 0 or 1.
            y (numpy.ndarray): Training labels of shape (num_samples,).
        """
        # Identify unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Initialize parameters
        self.priors_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_probs_ = np.zeros((n_classes, n_features), dtype=np.float64)

        # Calculate priors and feature probabilities
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.priors_[idx] = X_c.shape[0] / float(X.shape[0])  # P(c)

            # Sum over each feature (how many 1s in feature j for class c)
            # Laplace smoothing: theta_{c,j} = (count_of_ones + alpha) / (count_of_samples + 2*alpha)
            count_ones = np.sum(X_c, axis=0)
            self.feature_probs_[idx, :] = (count_ones + self.alpha) / (X_c.shape[0] + 2 * self.alpha)

    def predict(self, X):
        """
        Predicts class labels for samples in X.
        
        Args:
            X (numpy.ndarray): Binary test features of shape (num_samples, num_features).
        
        Returns:
            y_pred (numpy.ndarray): Predicted labels of shape (num_samples,).
        """
        # Compute predictions for each sample
        y_pred = [self._predict_single(sample) for sample in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        """
        Computes the predicted class for a single binary feature vector x.
        
        Args:
            x (numpy.ndarray): Binary feature vector of shape (num_features,).
        
        Returns:
            c (int or float): Predicted class label.
        """
        posteriors = []
        for idx, c in enumerate(self.classes_):
            # log P(c)
            prior_log = np.log(self.priors_[idx])
            
            # Bernoulli log-likelihood:
            # sum_{j=1 to d} [ x_j * log(theta_{c,j}) + (1 - x_j) * log(1 - theta_{c,j}) ]
            theta_c = self.feature_probs_[idx]
            # Clip values to avoid log(0)
            # (Although with alpha smoothing, we shouldn’t get zero, but just to be safe.)
            theta_c = np.clip(theta_c, 1e-12, 1 - 1e-12)
            
            log_likelihood = np.sum(x * np.log(theta_c) + (1 - x) * np.log(1 - theta_c))
            
            posterior = prior_log + log_likelihood
            posteriors.append(posterior)

        # Return class with the highest posterior
        return self.classes_[np.argmax(posteriors)]

def preprocess_data(train_dataset, test_dataset):
    """
    Preprocess data for Bernoulli Naive Bayes.
    Flattens images, normalizes, and binarizes (threshold at 0.5).
    """
    # Extract images and labels
    X_train = train_dataset.imgs
    y_train = train_dataset.labels
    X_test = test_dataset.imgs
    y_test = test_dataset.labels
    
    # Flatten images and convert to float
    X_train_float = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_test_float = X_test.reshape(X_test.shape[0], -1).astype(np.float32)
    
    # Normalize pixel values to [0,1]
    X_train_float /= 255.0
    X_test_float /= 255.0
    
    # Binarize the normalized data (threshold at 0.5)
    X_train_processed = (X_train_float > 0.5).astype(np.int32)
    X_test_processed = (X_test_float > 0.5).astype(np.int32)
    
    # Ensure labels are 1D arrays
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    
    return X_train_processed, y_train, X_test_processed, y_test

def run_bernoulli_naive_bayes(train_dataset, test_dataset):
    """
    Train and evaluate a Bernoulli Naive Bayes model .
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        
    Returns:
        accuracy: Accuracy score on test set
        y_pred: Predictions on test set
        y_test: True labels for test set
        model: The trained Bernoulli Naive Bayes model
    """
    # Preprocess data
    X_train, y_train, X_test, y_test = preprocess_data(train_dataset, test_dataset)
    
    print("\n" + "="*80)
    print("Running Bernoulli Naive Bayes model ()...")
    print("="*80)
    
    # Instantiate and train the model
    model = MyBernoulliNaiveBayes(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=['Normal', 'Pneumonia'])
    
    print(f"Bernoulli NB Results on Test Data ():")
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
        save_path: Path to save the visualization
    """
    cm = confusion_matrix(actual_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Normal", "Pneumonia"], 
                yticklabels=["Normal", "Pneumonia"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Bernoulli Naive Bayes - Confusion Matrix")
    
    # Save the visualization if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()
