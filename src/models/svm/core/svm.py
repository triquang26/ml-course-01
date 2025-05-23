import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

def run_svm(train_dataset, test_dataset, kernel='rbf', C=1.0, gamma='scale'):
    """
    Run a Support Vector Machine model on the PneumoniaMNIST dataset.
    
    Args:
        train_dataset: Training dataset from MedMNIST
        test_dataset: Test dataset from MedMNIST
        kernel: Kernel type for SVM ('linear', 'poly', 'rbf', 'sigmoid')
        C: Regularization parameter
        gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        
    Returns:
        accuracy: Model accuracy on test set
        predictions: Model predictions on test set
        actual_labels: True labels for test set
        model: Trained SVM model
        preprocessors: Dictionary of SVM parameters used
    """
    # Data preparation
    print("Preparing data for SVM model...")
    x_train = train_dataset.imgs.astype('float32') / 255.0
    y_train = train_dataset.labels.flatten()

    x_test = test_dataset.imgs.astype('float32') / 255.0
    y_test = test_dataset.labels.flatten()

    # Flatten images
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    print(f"Training data shape: {x_train_flat.shape}")
    print(f"Test data shape: {x_test_flat.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")

    # Handle large datasets by subsampling if needed
    max_train_samples = 5000  # Limit for SVM training to manage memory and time
    if len(x_train_flat) > max_train_samples:
        print(f"Sampling {max_train_samples} examples from {len(x_train_flat)} for training")
        idx = np.random.choice(len(x_train_flat), max_train_samples, replace=False)
        x_train_subset = x_train_flat[idx]
        y_train_subset = y_train[idx]
    else:
        x_train_subset = x_train_flat
        y_train_subset = y_train

    # Train SVM model
    start_time = time.time()
    print("\nTraining SVM model...")
    
    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
    model.fit(x_train_subset, y_train_subset)
    
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")

    # Evaluate on test set
    print("Making predictions on test data...")
    start_time = time.time()
    
    # For large test sets, use batching
    max_test_samples = 10000
    if len(x_test_flat) > max_test_samples:
        print(f"Using first {max_test_samples} samples for testing")
        x_test_subset = x_test_flat[:max_test_samples]
        y_test_subset = y_test[:max_test_samples]
    else:
        x_test_subset = x_test_flat
        y_test_subset = y_test
    
    predictions = model.predict(x_test_subset)
    
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.2f} seconds for {len(x_test_subset)} samples")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_subset, predictions)
    print(f"\nAccuracy on test data: {accuracy:.4f}")
    
    # Return results
    preprocessors = {
        'params': {
            'kernel': kernel,
            'C': C,
            'gamma': gamma
        }
    }
    
    return accuracy, predictions, y_test_subset, model, preprocessors

def visualize_results(predictions, actual_labels, save_path=None):
    """Visualize the performance of the SVM with confusion matrix and classification report
    
    Args:
        predictions: Model predictions on test set
        actual_labels: True labels for test set
        save_path: Path to save the confusion matrix visualization (optional)
    """
    cm = confusion_matrix(actual_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Support Vector Machine')
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()

    print("\nClassification Report:")
    print(classification_report(actual_labels, predictions, 
                              target_names=['Normal', 'Pneumonia']))

    print("\nSVM analysis completed!")