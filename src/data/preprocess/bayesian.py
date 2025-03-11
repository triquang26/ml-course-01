import os
import time
import sys
from medmnist import INFO, PneumoniaMNIST
import matplotlib.pyplot as plt
import numpy as np
def load_data():
    """Load the PneumoniaMNIST dataset"""
    print("Loading PneumoniaMNIST dataset...")
    
    # Load dataset information
    dataset_info = INFO["pneumoniamnist"]
    print(f"Dataset description: {dataset_info['description']}")
    print(f"Number of classes: {len(dataset_info['label'])}, Labels: {dataset_info['label']}")
    
    # Load training and test sets
    train_dataset = PneumoniaMNIST(split='train', download=True)
    test_dataset = PneumoniaMNIST(split='test', download=True)
    
    print("Dataset loaded successfully.")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset


def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")