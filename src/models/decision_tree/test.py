import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.data.preprocess.decision_tree import load_data
from predict import PneumoniaPredictor

def visualize_results(y_true, y_pred, save_path=None):
    """Visualize test results"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                    horizontalalignment='center',
                    verticalalignment='center')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Results visualization saved to: {save_path}")
    plt.close()

def test_model():
    """Test the trained model"""
    # Load test data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Initialize predictor
    predictor = PneumoniaPredictor()
    
    # Make predictions
    predictions, probabilities = predictor.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    print("\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Visualize results
    visualize_results(
        y_test, 
        predictions, 
        save_path="figures/test_results_confusion_matrix.png"
    )

    return accuracy, predictions, y_test

if __name__ == "__main__":
    test_model()