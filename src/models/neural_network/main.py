import os
import time
from medmnist import INFO, PneumoniaMNIST
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

# Import the Vision Transformer implementation
from neural_network import (
    run_vision_transformer, 
    visualize_results, 
    compute_mean_std, 
    plot_confusion_matrix,
    plot_f1_heatmap
)

# Define visualization directory
VISUALIZATION_DIR = "reports/figures"

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

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def main():
    """Main function to run Vision Transformer model"""
    # Create visualization directory if it doesn't exist
    ensure_dir_exists(VISUALIZATION_DIR)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Load data
    train_dataset, test_dataset = load_data()
    
    # Track results
    results = {}
    
    # Set training parameters
    n_epochs = 1
    batch_size = 100
    learning_rate = 0.003
    
    print(f"\n{'='*80}")
    print("TRAINING VISION TRANSFORMER MODEL")
    print(f"{'='*80}")
    print(f"Epochs: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    start_time = time.time()
    
    # Run Vision Transformer
    vit_accuracy, vit_predictions, vit_actual_labels, vit_model = run_vision_transformer(
        train_dataset,
        test_dataset,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save results
    visualize_results(
        vit_predictions,
        vit_actual_labels,
        save_path=os.path.join(VISUALIZATION_DIR, f"{timestamp}_vision_transformer_cm.png")
    )
    
    # Generate additional visualizations
    plot_confusion_matrix(
        vit_model,
        torch.utils.data.DataLoader(test_dataset, batch_size=1000),
        save_path=os.path.join(VISUALIZATION_DIR, f"{timestamp}_vision_transformer_detailed_cm.png")
    )
    
    plot_f1_heatmap(
        vit_model,
        torch.utils.data.DataLoader(test_dataset, batch_size=1000),
        save_path=os.path.join(VISUALIZATION_DIR, f"{timestamp}_vision_transformer_f1_heatmap.png")
    )
    
    # Store result
    results['Vision Transformer'] = vit_accuracy
    
    # Display results
    print("\n" + "="*80)
    print("MODEL RESULTS")
    print("="*80)
    print(f"Vision Transformer: Accuracy = {vit_accuracy:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.title('Model Performance - Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    for i, (model, acc) in enumerate(results.items()):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    # Save the comparison plot
    comparison_path = os.path.join(VISUALIZATION_DIR, f"{timestamp}_vision_transformer_performance.png")
    plt.savefig(comparison_path)
    print(f"Saved performance plot to {comparison_path}")
    plt.close()
    
    print("\nModel summary:")
    print(f"- Architecture: Vision Transformer")
    print(f"- Parameters: {sum(p.numel() for p in vit_model.parameters() if p.requires_grad)}")
    print(f"- Training time: {training_time:.2f} seconds")
    print(f"- Test accuracy: {vit_accuracy:.4f}")

if __name__ == "__main__":
    main()