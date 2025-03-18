import numpy as np
import torch
from medmnist import PneumoniaMNIST
from sklearn.model_selection import train_test_split
from medmnist import INFO, PneumoniaMNIST

def load_data():
    """Load and preprocess the PneumoniaMNIST dataset"""
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
    
    # Preprocess images
    x_train = train_dataset.imgs.astype('float32') / 255.0
    y_train = train_dataset.labels.flatten()
    
    x_test = test_dataset.imgs.astype('float32') / 255.0
    y_test = test_dataset.labels.flatten()
    
    # Flatten images for traditional models
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    
    # Convert to tensor format for CNN
    x_train_tensor = numpy_to_tensor(x_train)
    x_test_tensor = numpy_to_tensor(x_test)
    
    # Split training data to create validation set for GA optimization
    print("Creating validation split...")
    (x_train_flat_model, x_val_flat, 
     x_train_tensor_model, x_val_tensor, 
     y_train_model, y_val) = train_test_split(
        x_train_flat, x_train_tensor, y_train, test_size=0.2, random_state=42)
    
    print(f"Model training samples: {len(y_train_model)}")
    print(f"Validation samples: {len(y_val)}")
    
    return (x_train_flat_model, x_val_flat, x_test_flat, 
            x_train_tensor_model, x_val_tensor, x_test_tensor,
            y_train_model, y_val, y_test)

def numpy_to_tensor(x):
    """
    Converts a NumPy array to a PyTorch tensor, adding a channel dimension if necessary.
    """
    if x.ndim == 3:
        x = np.expand_dims(x, axis=1)
    elif x.ndim == 4 and x.shape[-1] == 1:
        x = x.transpose(0, 3, 1, 2)
    return torch.tensor(x, dtype=torch.float32)

def split_data(x, y, test_size=0.2, random_state=42):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)
