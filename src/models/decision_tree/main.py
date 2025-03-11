import os
import time
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from medmnist import PneumoniaMNIST
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from decision_tree import run_decision_tree_model, visualize_results


from src.data.preprocess.decision_tree import (
    load_data,
    ensure_dir_exists,
    compute_mean_std
)

from src.visualization.visualize import (
    visualize_results,
    compare_models,
    VISUALIZATION_DIR
)

from persistence import ModelPersistence

def main():
    """Main function to run the decision tree model"""
    # Create visualization directory if it doesn't exist
    ensure_dir_exists(VISUALIZATION_DIR) 
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Initialize model persistence
    model_persistence = ModelPersistence()
    
    # Set model parameters
    max_depth = 10  # Maximum depth of decision tree
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Track results for comparison
    results = {}
    
    # Run Decision Tree model
    print("\n" + "="*80)
    print("Running Decision Tree model...")
    print("="*80)
    
    # Update to capture the returned model
    dt_accuracy, dt_predictions, dt_actual_labels, dt_model = run_decision_tree_model(
        X_train, y_train, X_val, y_val, X_test, y_test, max_depth=max_depth
    )
    
    # Save the trained model
    model_name = f"decision_tree_{timestamp}.joblib"
    model_path = model_persistence.save_model(dt_model, model_name)
    print(f"\nModel saved to: {model_path}")
    
    # Optional: Test loading the model
    loaded_model = model_persistence.load_model(model_path)
    print("Successfully loaded the model")
    
    visualize_results(
        dt_predictions,
        dt_actual_labels,
        model_name="Decision Tree",
        save_path=os.path.join(VISUALIZATION_DIR, f"{timestamp}_decision_tree_cm.png")
    )
    
    results['Decision Tree'] = dt_accuracy
if __name__ == "__main__":
    main()