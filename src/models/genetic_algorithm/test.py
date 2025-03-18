import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.models.genetic_algorithm.core.cnn_model import SimpleCNN
from src.data.preprocess.genetic_algorithm import load_data, numpy_to_tensor
from src.models.genetic_algorithm.persistance import ModelPersistence
from src.visualization.visualize import VISUALIZATION_DIR


def preprocess_data(test_dataset):
    """Preprocess test data using project pipeline"""
    # Extract data and labels
    X_test = test_dataset.imgs
    y_test = test_dataset.labels.squeeze()
    
    # Create flattened version for traditional models
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_test_flat = X_test_flat.astype('float32') / 255.0
    
    # Create tensor version for CNN
    X_test_tensor = numpy_to_tensor(X_test)
    
    return X_test_flat, X_test_tensor, y_test

def test_models():
    """Test all trained models"""
    # Load test dataset
    (x_train_flat, x_val_flat, x_test_flat,
     x_train_tensor, x_val_tensor, x_test_tensor,
     y_train, y_val, y_test) = load_data()
    
    # Preprocess test data is now handled by load_data()
    X_test_flat = x_test_flat
    X_test_tensor = x_test_tensor
    y_test = y_test
    
    # Rest of the function remains the same...
    # Load trained models
    model_persistence = ModelPersistence()
    models = model_persistence.load_models()
    
    if not models:
        print("No trained models found. Please train the models first.")
        return
    
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\nTesting individual models...")
    
    # Test Decision Tree
    if 'decision_tree' in models:
        print("\nTesting Decision Tree...")
        dt_probs = models['decision_tree'].predict_proba(X_test_flat)[:, 1]
        dt_preds = (dt_probs > 0.5).astype(int)
        results['Decision Tree'] = calculate_metrics(y_test, dt_preds)
    
    # Test Naive Bayes
    if 'naive_bayes' in models:
        print("\nTesting Naive Bayes...")
        nb_probs = models['naive_bayes'].predict_proba(X_test_flat)[:, 1]
        nb_preds = (nb_probs > 0.5).astype(int)
        results['Naive Bayes'] = calculate_metrics(y_test, nb_preds)
    
    # Test CNN
    if 'cnn_weights' in models:
        print("\nTesting CNN...")
        cnn_model = SimpleCNN()  # Create new CNN model instance
        cnn_model.load_state_dict(models['cnn_weights'])  # Load saved weights
        cnn_model.to(device)
        cnn_model.eval()
        with torch.no_grad():
            X_test_device = X_test_tensor.to(device)
            cnn_probs = cnn_model(X_test_device).cpu().numpy().flatten()
        cnn_preds = (cnn_probs > 0.5).astype(int)
        results['CNN'] = calculate_metrics(y_test, cnn_preds)
    
    # Test Ensemble with GA weights
    if all(k in models for k in ['decision_tree', 'naive_bayes', 'cnn_weights', 'ga_weights']):
        print("\nTesting GA Ensemble...")
        # Get probabilities from all models
        probs = [
            dt_probs,
            cnn_probs,
            nb_probs
        ]
        
        # Apply GA weights for ensemble prediction
        ensemble_probs = np.zeros_like(dt_probs)
        for weight, prob in zip(models['ga_weights'], probs):
            ensemble_probs += weight * prob
            
        ensemble_preds = (ensemble_probs > 0.5).astype(int)
        results['GA Ensemble'] = calculate_metrics(y_test, ensemble_preds)
    
    # Print results
    print("\nTest Results:")
    print("=" * 80)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
        print("-" * 40)
    
    # Visualize results
    visualize_test_results(results)

def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics for a model"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

def visualize_test_results(results):
    """Create visualizations for test results"""
    # Ensure visualization directory exists
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    # Create comparison plot
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        plt.bar(x + i*width, values, width, label=metric.capitalize())
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width*1.5, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save comparison plot
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'test_results_comparison.png'))
    plt.close()
    
    # Create confusion matrix plots
    for model_name, metrics in results.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Pneumonia'],
                   yticklabels=['Normal', 'Pneumonia'])
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save confusion matrix plot
        plt.savefig(os.path.join(VISUALIZATION_DIR, f'test_results_{model_name.lower().replace(" ", "_")}_cm.png'))
        plt.close()

if __name__ == "__main__":
    test_models()