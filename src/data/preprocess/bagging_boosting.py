import os
from medmnist import INFO, PneumoniaMNIST


def load_data(download=True):
    """Load the PneumoniaMNIST dataset for bagging and boosting experiments."""
    print("Loading PneumoniaMNIST dataset...")

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
