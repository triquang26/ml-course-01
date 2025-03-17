import torch
from torch import nn
from torch import nn, einsum
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
import torchvision
import time
from torchinfo import summary
from torchvision import datasets, transforms
import torch
from torch import nn
from torch import nn, einsum
import torch.nn.functional as F
from torch import optim

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import torchvision
import time
from torchinfo import summary
from medmnist import PneumoniaMNIST
from medmnist import INFO

def load_data():
    """Load and preprocess the PneumoniaMNIST dataset"""
    print("Loading PneumoniaMNIST dataset...")
    
    # Load dataset information
    dataset_info = INFO["pneumoniamnist"]
    print(f"Dataset description: {dataset_info['description']}")
    print(f"Number of classes: {len(dataset_info['label'])}, Labels: {dataset_info['label']}")
    
    # First load the raw dataset to compute mean and std
    raw_train_dataset = PneumoniaMNIST(split='train', download=True, transform=transforms.ToTensor())
    mean, std = compute_mean_std(raw_train_dataset)
    print(f"Dataset statistics - Mean: {mean:.4f}, Std: {std:.4f}")
    
    # Create transform with normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])
    
    # Load training and test sets with proper transform
    train_dataset = PneumoniaMNIST(split='train', download=True, transform=transform)
    test_dataset = PneumoniaMNIST(split='test', download=True, transform=transform)
    
    print("Dataset loaded successfully.")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset

def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=500, shuffle=False, num_workers=2)
    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (number of images in a batch)
        images = images.view(batch_samples, -1)  # Flatten to (batch_size, H*W)

        mean += images.mean(dim=1).sum().item()
        std += images.std(dim=1).sum().item()
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean, std
