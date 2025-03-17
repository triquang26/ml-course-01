import os
import time
from medmnist import INFO, PneumoniaMNIST
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import os
import sys
import time
from typing import Dict, Tuple, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# Import the Vision Transformer implementation
from neural_network import (
    run_vision_transformer, 
    test_vision_transformer,
    # visualize_results, 
    compute_mean_std, 
    plot_confusion_matrix,
    plot_f1_heatmap
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.data.preprocess.neural_network import *
from src.features.features_selection.neural_network import *
from src.visualization.visualize import (
    visualize_results,
    compare_models,
    VISUALIZATION_DIR
)
MODEL_DIR = "trained"
# Define visualization directory
VISUALIZATION_DIR = "reports/figures"
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    
def main():
    """Main function to run Vision Transformer model"""
    # Create visualization directory if it doesn't exist
    ensure_dir_exists(VISUALIZATION_DIR)
    ensure_dir_exists(MODEL_DIR)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results = {}
    # Load data
    train_dataset, test_dataset = load_data()
    
    # Track results
    # results = {}
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    data_sample = next(iter(train_loader))[0]
    C, H, W = data_sample.shape[1], data_sample.shape[2], data_sample.shape[3]
    
    # ViT hyperparameters
    patch = 4  # Patch size (image dimensions must be divisible by this)
    F_out = len(np.unique(train_dataset.labels))  # Number of classes
    F_o = 64  # Output dimension in attention mechanism
    heads = 4  # Number of attention heads
    F_in = 64  # Input dimension to transformer
    mlp_dim = 128  # Hidden dimension in feed-forward network
    trans_depth = 6
    # Set training parameters
  
    start_time = time.time()
    
    # Run Vision Transformer
    # vit_model = VIT(1, 28, 28, 10, F_in, F_o, F_out, heads, trans_depth, mlp_dim)
    vit_model = VIT(
        channel=C,
        image_h=H,
        image_w=W,
        patch=patch,
        F_in=F_in,
        F_o=F_o,
        F_out=F_out,
        heads=heads,
        trans_depth=trans_depth,
        mlp_dim=mlp_dim,
        mode="zero"  # Use CLS token for classification
    )
    state_dict = torch.load("./trained/ViT.pth")
    vit_model.load_state_dict(state_dict)
    # print(vit_model)
    vit_model.eval()
    vit_accuracy, vit_predictions, vit_actual_labels = test_vision_transformer(
        test_dataset,
        vit_model
    )
    visualize_results(
        vit_predictions,
        vit_actual_labels,
        "ViT",
        save_path=os.path.join(VISUALIZATION_DIR, f"{timestamp}_test_vision_transformer_cm.png")
    )
    plot_confusion_matrix(
    vit_predictions,
    vit_actual_labels,
    save_path=os.path.join(VISUALIZATION_DIR, f"{timestamp}_test_vision_transformer_detailed_cm.png")
    )
    plot_f1_heatmap(
    vit_predictions,
    vit_actual_labels,
        save_path=os.path.join(VISUALIZATION_DIR, f"{timestamp}_test_vision_transformer_f1_heatmap.png")
    )
    metrics = {
    'accuracy': accuracy_score(vit_actual_labels, vit_predictions),
    'precision': precision_score(vit_actual_labels, vit_predictions, average='weighted'),
    'recall': recall_score(vit_actual_labels, vit_predictions, average='weighted'),
    'f1_score': f1_score(vit_actual_labels, vit_predictions, average='weighted')
    }
    results['ViT'] = metrics
    evaluation_time = time.time() - start_time
    print(f"Evaluation completed in {evaluation_time:.2f} seconds")
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    print(f"\n VISION TRANSFORMER:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print("-" * 40)

if __name__ == "__main__":
    main()