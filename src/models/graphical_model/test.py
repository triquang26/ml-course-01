import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.data.preprocess.graphical_model import (
    load_data,
    extract_data_labels,
    discretize_features,
    ensure_dir_exists
)
from src.features.features_selection.graphical_model import (
    apply_pca,
    calculate_feature_correlation
)
from src.models.graphical_model.predict import ModelPredictor
from src.visualization.visualize import visualize_results, VISUALIZATION_DIR

def preprocess_data(test_dataset):
    """Preprocess test data using project pipeline"""
    # Extract data and labels
    X_test = test_dataset.imgs
    y_test = test_dataset.labels.squeeze()
    
    # Basic preprocessing
    X_test = X_test.reshape(X_test.shape[0], -1)  # Flatten images
    X_test = X_test.astype('float32') / 255.0  # Normalize
    
    # Add PCA dimensionality reduction to match model's feature count
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50)  # Model expects 50 components
    X_test_reduced = pca.fit_transform(X_test)
    
    # Add discretization to match model's expected input format
    from sklearn.preprocessing import KBinsDiscretizer
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    X_test_discrete = discretizer.fit_transform(X_test_reduced)
    
    return X_test_discrete, y_test
def test_models():
    # Load test dataset
    _, test_dataset = load_data()
    
    # Get raw test data
    X_test, y_test = preprocess_data(test_dataset)
    
    # Initialize predictor with preprocessing components
    predictor = ModelPredictor()
    
    # Models to test
    model_names = [
        'bayesian_network',
        # 'augmented_naive_bayes',
        # 'hidden_markov_model'
    ]
    
    results = {}
    
    for model_name in model_names:
        print(f"\nTesting {model_name}...")
        
        # Load model and its preprocessing components
        if not predictor.load_model(model_name):
            print(f"Could not load {model_name} model. Skipping...")
            continue
            
        try:
            # Make predictions
            y_pred = predictor.predict(X_test, model_name)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
            
            results[model_name] = metrics
            
            # Visualize results
            visualize_results(
                y_true=y_test,
                y_pred=y_pred,
                model_name=model_name,
                save_path=os.path.join(VISUALIZATION_DIR, f"test_{model_name}_results.png")
            )
            
            # Plot confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(VISUALIZATION_DIR, f"test_{model_name}_confusion_matrix.png"))
            plt.close()
            
        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")
            continue

    # Print summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print("-" * 40)

if __name__ == "__main__":
    test_models()