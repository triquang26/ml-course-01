import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix

def run_decision_tree_model(X_train, y_train, X_val, y_val, X_test, y_test, max_depth=10):
    """
    Train and evaluate a decision tree model.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        max_depth: Maximum depth of the decision tree
        
    Returns:
        test_accuracy: Accuracy on test set
        y_test_pred: Predictions on test set
        y_test: True labels for test set
    """
    # Train the model
    dt_model = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Evaluation on validation set
    y_val_pred = dt_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    
    # Evaluation on test set
    y_test_pred = dt_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Calculate F1 scores
    f1 = f1_score(y_test, y_test_pred)
    f1_macro = f1_score(y_test, y_test_pred, average="macro")
    print(f"F1-score on Test Set: {f1:.4f}")
    print(f"Macro F1-score on Test Set: {f1_macro:.4f}")
    
    return test_accuracy, y_test_pred, y_test

def visualize_results(predictions, actual_labels, save_path=None):
    """
    Visualize the results of the model evaluation.
    
    Args:
        predictions: Model predictions
        actual_labels: True labels
        save_path: Path to save the visualization
    """
    # Print classification report
    print("Classification Report on Test Set:")
    print(classification_report(actual_labels, predictions))
    
    # Create confusion matrix
    cm = confusion_matrix(actual_labels, predictions)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Normal", "Pneumonia"], 
                yticklabels=["Normal", "Pneumonia"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    
    # Save the visualization if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()