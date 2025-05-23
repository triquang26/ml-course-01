import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.data.preprocess.conditional_random_field import load_data
from src.models.conditional_random_field.persistance import ModelPersistence
from src.visualization.visualize import VISUALIZATION_DIR
from predict import ModelPredictor
from conditional_random_field import ConditionalRandomField

def test_models():
    """Test CRF model"""
    # Load test dataset
    print("\nLoading data...")
    (_, _, X_test_crf, Y_test_crf) = load_data()
    
    # Load trained model
    print("\nLoading trained model...")
    model_persistence = ModelPersistence()
    crf_model = model_persistence.load_model()
    
    if not crf_model:
        print("No trained model found. Please train the model first.")
        return
    
    # Create predictor
    predictor = ModelPredictor(ConditionalRandomField(model_file=os.path.join(model_persistence.base_path, 'crf_model.crfsuite')))
    
    print("\nMaking predictions...")
    predictions = predictor.predict(X_test_crf)
    
    # Prepare labels for evaluation
    y_true = [y[0] for y in Y_test_crf]  # Extract single labels from lists
    y_pred = [p[0] for p in predictions]  # Extract single predictions from lists
    
    # Calculate metrics
    results = {
        'Conditional Random Field': {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, pos_label='[1]'),
            'recall': recall_score(y_true, y_pred, pos_label='[1]'),
            'f1_score': f1_score(y_true, y_pred, pos_label='[1]'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    }
    
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