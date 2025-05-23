import os
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from persistance import ModelPersistence
from conditional_random_field import ConditionalRandomField
from predict import ModelPredictor

persistant = ModelPersistence()
MODEL_FILENAME = os.path.join(persistant.base_path, 'crf_model.crfsuite')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.data.preprocess.conditional_random_field import load_data
from src.models.conditional_random_field.model_tuning import train_crf

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
    
    print(f"{'='*80}\nCONDITIONAL RANDOM FIELD\n{'='*80}")
    
    # Load and preprocess data
    (X_train_crf, Y_train_crf, X_test_crf, Y_test_crf) = load_data()
    
    crf = ConditionalRandomField(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
        model_file=MODEL_FILENAME
    )

    crf_model = train_crf(X_train_crf, Y_train_crf, crf)     

    # Evaluate individual models on test set
    print("\nEvaluating models on test set...")
    predictor = ModelPredictor(crf_model)
    
    model_names = ["Conditional Random Field"]
    test_preds = [predictor.predict(X_test_crf)]
    
    accuracies = []
    f1_scores = []
    confusion_matrices = []
    
    for name, preds in zip(model_names, test_preds):
        # Flatten the nested lists for evaluation
        y_true = [y[0] for y in Y_test_crf]  # Extract single labels from lists
        y_pred = [p[0] for p in preds]  # Extract single predictions from lists
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, pos_label='[1]')  # Changed from '1' to '[1]'
        cm = confusion_matrix(y_true, y_pred)
        
        accuracies.append(acc)
        f1_scores.append(f1)
        confusion_matrices.append(cm)
        
        print(f"\nRESULTS:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(model_names, accuracies, f1_scores, confusion_matrices, timestamp)

    print("\nComplete!")

if __name__ == "__main__":
    main()