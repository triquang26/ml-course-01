import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from medmnist import PneumoniaMNIST
from sklearn.model_selection import train_test_split

def compute_mean_std(dataset):
    """Compute the mean and standard deviation of a dataset."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False)
    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, -1)
        mean += images.mean(dim=1).sum().item()
        std += images.std(dim=1).sum().item()
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean, std

def load_data():
    """Load and preprocess the PneumoniaMNIST dataset."""
    print("Loading PneumoniaMNIST dataset...")
    # Get the raw dataset to compute statistics
    raw_dataset = PneumoniaMNIST(split='train', download=True, transform=transforms.ToTensor())
    mean, std = compute_mean_std(raw_dataset)
    print(f"Mean: {mean}, Std: {std}")

    # Apply normalization transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    # Load datasets with transform
    train_dataset = PneumoniaMNIST(split='train', download=True, transform=transform)
    test_dataset = PneumoniaMNIST(split='test', download=True, transform=transform)
    
    print("Dataset loaded successfully.")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Convert to NumPy arrays
    X_train = np.array([img.numpy().flatten() for img, _ in train_dataset])
    y_train = np.array([label for _, label in train_dataset]).flatten()

    X_test = np.array([img.numpy().flatten() for img, _ in test_dataset])
    y_test = np.array([label for _, label in test_dataset]).flatten()

    # Split training data to create validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")