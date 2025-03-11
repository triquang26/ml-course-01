import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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
    Train and evaluate a Gaussian Naive Bayes model.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        
    Returns:
        accuracy: Accuracy score on test set
        y_pred: Predictions on test set
        y_test: True labels for test set
    """
    # Preprocess data
    X_train, y_train, X_test, y_test = preprocess_data(train_dataset, test_dataset)
    
    print("\n" + "="*80)
    print("Running Gaussian Naive Bayes model...")
    print("="*80)
    
    # Train the model
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    
    # Make predictions
    y_pred = gnb.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=['Normal', 'Pneumonia'])
    
    print(f"Gaussian NB Results on Test Data:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    
    return accuracy, y_pred, y_test

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
    plt.title("Gaussian Naive Bayes - Confusion Matrix")
    
    # Save the visualization if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()