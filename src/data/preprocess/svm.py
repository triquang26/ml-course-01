import os
from medmnist import INFO, PneumoniaMNIST


def load_data(download=True):
    """Load the PneumoniaMNIST dataset for SVM experiments."""
    print("Loading PneumoniaMNIST dataset for SVM...")

    # Print dataset metadata
    dataset_info = INFO['pneumoniamnist']
    print(f"Dataset description: {dataset_info['description']}")
    print(f"Number of classes: {len(dataset_info['label'])}, Labels: {dataset_info['label']}")

    # Download and split data
    train_dataset = PneumoniaMNIST(split='train', download=download)
    test_dataset = PneumoniaMNIST(split='test', download=download)

    print("Dataset loaded successfully.")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_dataset, test_dataset


def ensure_dir_exists(directory):
    """Ensure that a directory exists; create it if not."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def extract_data_labels(dataset):
    """Extract features and labels from dataset for SVM processing."""
    X = dataset.imgs.astype('float32') / 255.0
    y = dataset.labels.flatten()
    
    # Flatten images for SVM
    X_flat = X.reshape(X.shape[0], -1)
    
    print(f"Extracted data shape: {X_flat.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes: {len(set(y))}")
    
    return X_flat, y


def preprocess_for_svm(X, y=None):
    """
    Preprocess data specifically for SVM training/prediction
    
    Args:
        X: Input features (images)
        y: Labels (optional, for training data)
        
    Returns:
        Preprocessed features and labels (if provided)
    """
    # Ensure data is properly normalized
    if X.max() > 1.0:
        X = X.astype('float32') / 255.0
    
    # Flatten if needed
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    
    print(f"SVM preprocessing complete. Data shape: {X.shape}")
    
    if y is not None:
        return X, y
    return X