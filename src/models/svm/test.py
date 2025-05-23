import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.data.preprocess.svm import load_data, preprocess_for_svm
from src.models.svm.predict import ModelPredictor
from src.visualization.visualize import visualize_results, VISUALIZATION_DIR
from src.models.svm.persistence import ModelPersistence

def preprocess_data(test_dataset):
    """Preprocess test data for SVM"""
    # Extract data and labels
    X_test = test_dataset.imgs
    y_test = test_dataset.labels.squeeze()
    
    # Apply SVM preprocessing
    X_test_processed = preprocess_for_svm(X_test)
    
    return X_test_processed, y_test

def test_svm_model():
    """Test the SVM model"""
    # Load test dataset
    _, test_dataset = load_data()
    
    # Get preprocessed test data
    X_test, y_test = preprocess_data(test_dataset)
    
    # Initialize predictor
    predictor = ModelPredictor()
    
    # Model to test
    model_name = 'support_vector_machine'
    
    print(f"Testing {model_name}...")
    
    # Load model
    if not predictor.load_model(model_name):
        print(f"Could not load {model_name} model. Skipping...")
        return
        
    try:
        # Make predictions
        y_pred = predictor.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Visualize results
        visualize_results(
            y_true=y_test,
            y_pred=y_pred,
            model_name=model_name,
            save_path=os.path.join(VISUALIZATION_DIR, f"test_{model_name}_results.png")
        )
        
        # Print results
        print("\n" + "="*60)
        print("SVM TEST RESULTS")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print("-" * 60)
        
    except Exception as e:
        print(f"Error testing {model_name}: {str(e)}")

if __name__ == "__main__":
    test_svm_model()