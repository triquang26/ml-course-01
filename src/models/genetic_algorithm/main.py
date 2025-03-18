import os
import time
import numpy as np
import torch
from core import cnn_model, decision_tree_model, bayesian_model
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from medmnist import INFO, PneumoniaMNIST
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from persistance import ModelPersistence
from genetic_algorithm import GeneticAlgorithm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.data.preprocess.genetic_algorithm import (
    load_data,
    numpy_to_tensor
)

from src.models.genetic_algorithm.model_tuning import (
    train_cnn,
    train_decision_tree,
    train_naive_bayes
)

# Define visualization directory
VISUALIZATION_DIR = "reports/figures"

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def evaluate_model(model, x_test, y_test, model_name):
    """Evaluate model performance"""
    if model_name == "CNN":
        # Handle CNN evaluation separately because it uses tensors
        device = x_test.device if hasattr(x_test, 'device') else torch.device('cpu')
        model.eval()
        with torch.no_grad():
            outputs = model(x_test).cpu().numpy().flatten()
        predictions = (outputs > 0.5).astype(int)
    else:
        # For scikit-learn models
        predictions = model.predict(x_test)
    
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    return accuracy, f1, cm, predictions

def get_probabilities(model, x_test, model_name, device=None):
    """Get probability predictions from a model"""
    if model_name == "CNN":
        # Handle CNN separately because it uses tensors
        model.eval()
        with torch.no_grad():
            x_test_device = x_test.to(device) if device else x_test
            prob = model(x_test_device).cpu().numpy().flatten()
    else:
        # For scikit-learn models
        prob = model.predict_proba(x_test)[:, 1]  # Get probability of positive class
    
    return prob

def visualize_results(model_names, accuracies, f1_scores, confusion_matrices, timestamp):
    """Visualize model comparison results"""
    # Create bar plot for accuracy and F1 scores
    plt.figure(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(model_names))
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy')
    plt.bar(x + width/2, f1_scores, width, label='F1 Score')
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Comparison - Accuracy and F1 Score')
    plt.xticks(x, model_names)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add text labels
    for i, acc in enumerate(accuracies):
        plt.text(i - width/2, acc + 0.01, f'{acc:.4f}', ha='center')
    for i, f1 in enumerate(f1_scores):
        plt.text(i + width/2, f1 + 0.01, f'{f1:.4f}', ha='center')
    
    # Save plot
    comparison_path = os.path.join(VISUALIZATION_DIR, f"{timestamp}_model_comparison.png")
    plt.savefig(comparison_path)
    print(f"Saved comparison plot to {comparison_path}")
    plt.close()
    
    # Create confusion matrix plots
    for i, (name, cm) in enumerate(zip(model_names, confusion_matrices)):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=["Normal", "Pneumonia"],
                   yticklabels=["Normal", "Pneumonia"])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{name} - Confusion Matrix')
        
        cm_path = os.path.join(VISUALIZATION_DIR, f"{timestamp}_{name.lower().replace(' ', '_')}_cm.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"Saved {name} confusion matrix to {cm_path}")

def main():
    """Main function to run all models and ensemble"""
    # Create visualization directory if it doesn't exist
    ensure_dir_exists(VISUALIZATION_DIR)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    print(f"{'='*80}\nENSEMBLE MODEL WITH GENETIC ALGORITHM OPTIMIZATION\n{'='*80}")
    
    # Load and preprocess data
    (x_train_flat, x_val_flat, x_test_flat,
     x_train_tensor, x_val_tensor, x_test_tensor,
     y_train, y_val, y_test) = load_data()
    
    # Train individual models
    dt_model = train_decision_tree(x_train_flat, y_train)
    bayes_model = train_naive_bayes(x_train_flat, y_train)
    cnn_model, device = train_cnn(x_train_tensor, y_train)
    
    print("\nEvaluating models on validation set...")
    
    # Get validation probabilities for GA optimization
    dt_val_probs = get_probabilities(dt_model, x_val_flat, "Decision Tree")
    bayes_val_probs = get_probabilities(bayes_model, x_val_flat, "Naive Bayes")
    cnn_val_probs = get_probabilities(cnn_model, x_val_tensor, "CNN", device)
    
    # Run genetic algorithm to find optimal ensemble weights
    print("\nOptimizing ensemble weights with genetic algorithm...")
    ga = GeneticAlgorithm(pop_size=20, generations=200)
    best_weights, best_fitness = ga.run_ga(
        [dt_val_probs, cnn_val_probs, bayes_val_probs],
        y_val
    )

    print("\nSaving trained models...")
    model_saver = ModelPersistence()
    save_success = model_saver.save_models(
        dt_model=dt_model,
        bayes_model=bayes_model,
        cnn_model=cnn_model,
        ga_weights=best_weights
    )
    
    if save_success:
        print("All models saved successfully!")
    else:
        print("Warning: There was an issue saving one or more models.")

    
    print(f"\nOptimized Ensemble Weights: {[round(w, 3) for w in best_weights]}")
    print(f"Validation Fitness: {best_fitness:.4f}")

    # Evaluate individual models on test set
    print("\nEvaluating models on test set...")
    
    dt_test_probs = get_probabilities(dt_model, x_test_flat, "Decision Tree")
    bayes_test_probs = get_probabilities(bayes_model, x_test_flat, "Naive Bayes")
    cnn_test_probs = get_probabilities(cnn_model, x_test_tensor, "CNN", device)
    
    # Make ensemble prediction with optimized weights
    ensemble_test_probs = ga.ensemble_probabilities(
        best_weights,
        [dt_test_probs, cnn_test_probs, bayes_test_probs]
    )
    ensemble_test_preds = (ensemble_test_probs > 0.5).astype(int)
    
    # Evaluate results
    dt_test_preds = (dt_test_probs > 0.5).astype(int)
    bayes_test_preds = (bayes_test_probs > 0.5).astype(int)
    cnn_test_preds = (cnn_test_probs > 0.5).astype(int)
    
    model_names = ["Decision Tree", "Naive Bayes", "CNN", "GA Ensemble"]
    test_preds = [dt_test_preds, bayes_test_preds, cnn_test_preds, ensemble_test_preds]
    
    accuracies = []
    f1_scores = []
    confusion_matrices = []
    
    for name, preds in zip(model_names, test_preds):
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        
        accuracies.append(acc)
        f1_scores.append(f1)
        confusion_matrices.append(cm)
        
        print(f"\n{name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(model_names, accuracies, f1_scores, confusion_matrices, timestamp)
    
    print("\nEnsemble weights:")
    print(f"  Decision Tree: {best_weights[0]:.3f}")
    print(f"  CNN: {best_weights[1]:.3f}")
    print(f"  Naive Bayes: {best_weights[2]:.3f}")
    
    print("\nComplete!")

if __name__ == "__main__":
    main()