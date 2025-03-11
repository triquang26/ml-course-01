import os
import numpy as np
from medmnist import PneumoniaMNIST
from sklearn.preprocessing import KBinsDiscretizer

def load_data():
    """Load and prepare the PneumoniaMNIST dataset"""
    print("Loading PneumoniaMNIST dataset...")
    train_dataset = PneumoniaMNIST(split='train', download=True)
    test_dataset = PneumoniaMNIST(split='test', download=True)
    
    print("Dataset loaded successfully.")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset

def extract_data_labels(dataset):
    """Extract images and labels from dataset"""
    images = dataset.imgs.reshape(len(dataset), -1)  # Flatten images
    labels = dataset.labels.squeeze()  # Remove extra dimensions
    return images, labels

def discretize_features(train_features, test_features, n_bins=10, strategy='uniform'):
    """Discretize continuous features into bins"""
    print(f"Discretizing features into {n_bins} bins using {strategy} strategy...")
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    
    # Fit on training data and transform both training and test data
    train_discrete = discretizer.fit_transform(train_features)
    test_discrete = discretizer.transform(test_features)
    
    # Convert to integer for graphical model compatibility
    train_discrete = train_discrete.astype(int)
    test_discrete = test_discrete.astype(int)
    
    print(f"Discretized data shape - Train: {train_discrete.shape}, Test: {test_discrete.shape}")
    return train_discrete, test_discrete

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")