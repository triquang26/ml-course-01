import os
import sys
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.data.preprocess.bayesian import load_data
from persistence import ModelPersistence
from predict import BayesianPredictor

class BayesianTester:
    """Class for testing Bayesian models"""
    
    def __init__(self, model_type='gaussian'):
        """Initialize tester with model type"""
        self.model_type = model_type
        self.predictor = BayesianPredictor(model_type=model_type)
    
    def preprocess_test_data(self, X_test):
        """Preprocess test data based on model type"""
        # Flatten images if needed
        if len(X_test.shape) > 2:
            X_test = X_test.reshape(X_test.shape[0], -1)
            
        if self.model_type == 'gaussian':
            # Normalize to [0,1] for Gaussian NB
            X_processed = X_test.astype(np.float32) / 255.0
            
        elif self.model_type == 'bernoulli':
            # Binarize for Bernoulli NB
            X_processed = ((X_test.astype(np.float32) / 255.0) > 0.5).astype(np.int32)
            
        elif self.model_type == 'multinomial':
            # Use raw counts for Multinomial NB
            X_processed = X_test.astype(np.int32)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        return X_processed
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        try:
            # Preprocess test data
            X_processed = self.preprocess_test_data(X_test)
            
            # Get predictions
            predictions, probabilities = self.predictor.predict(X_processed)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            
            return accuracy, predictions, y_test
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise

def visualize_test_results(y_true, y_pred, model_type, save_path=None):
    """Visualize test results using confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_type.capitalize()} NB - Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0, 1], ['Normal', 'Pneumonia'])
    plt.yticks([0, 1], ['Normal', 'Pneumonia'])
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                    horizontalalignment='center',
                    verticalalignment='center')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    plt.close()

def test_models():
    """Test all Bayesian models"""
    # Load test data
    _, test_dataset = load_data()
    
    # Get raw test data
    X_test = test_dataset.imgs
    y_test = test_dataset.labels.reshape(-1)
    
    # Model types to test
    model_types = ['gaussian', 'bernoulli', 'multinomial']
    results = {}
    
    for model_type in model_types:
        print(f"\nTesting {model_type.capitalize()} Naive Bayes model...")
        tester = BayesianTester(model_type)
        
        try:
            accuracy, predictions, actuals = tester.evaluate(X_test, y_test)
            
            print(f"\nTest Results for {model_type.capitalize()} NB:")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(actuals, predictions))
            
            # Visualize results
            save_path = f"reports/figures/test_{model_type}_nb_cm.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            visualize_test_results(
                actuals, 
                predictions,
                model_type,
                save_path=save_path
            )
            
            results[model_type] = accuracy
            
        except Exception as e:
            print(f"Error testing {model_type} model: {str(e)}")
    
    return results

if __name__ == "__main__":
    results = test_models()
    print("\nFinal Results Summary:")
    for model_type, accuracy in results.items():
        print(f"{model_type.capitalize()} NB Accuracy: {accuracy:.4f}")