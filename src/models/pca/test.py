import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from medmnist import PneumoniaMNIST

# Define directory
VISUALIZATION_DIR = "/content/figures"

def evaluate_model(model, X, y, dataset_name="Test"):
    """
    Purpose: Evaluate a model on a dataset and print metrics.
    Input:
        - model: Trained model.
        - X: Features for evaluation.
        - y: True labels.
        - dataset_name (str): Name of the dataset (e.g., 'Validation', 'Test').
    Output: Tuple (accuracy, predictions) - Accuracy score and predicted labels.
    """
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions)
    f1_macro = f1_score(y, predictions, average="macro")
    print(f"{dataset_name} Accuracy: {accuracy * 100:.2f}%")
    print(f"{dataset_name} F1-score: {f1:.4f}")
    print(f"{dataset_name} Macro F1-score: {f1_macro:.4f}")
    return accuracy, predictions

def visualize_confusion_matrix(y_true, y_pred, model_name="SVM with PCA", save_path=None):
    """
    Purpose: Generate and display a confusion matrix.
    Input:
        - y_true: True labels.
        - y_pred: Predicted labels.
        - model_name (str): Name of the model for the plot title.
        - save_path (str, optional): Path to save the plot.
    Output: None
    """
    print(f"Classification Report for {model_name}:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"]))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Pneumonia"],
                yticklabels=["Normal", "Pneumonia"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")

    if save_path:
        ensure_dir_exists(os.path.dirname(save_path))
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    plt.show()

def visualize_pca_variance(pca, save_path=None):
    """
    Purpose: Plot the cumulative explained variance ratio of PCA components.
    Input:
        - pca: Fitted PCA object.
        - save_path (str, optional): Path to save the plot.
    Output: None
    """
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance, marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("PCA Explained Variance Ratio")
    plt.grid(True)
    
    if save_path:
        ensure_dir_exists(os.path.dirname(save_path))
        plt.savefig(save_path)
        print(f"Saved PCA variance plot to {save_path}")
    plt.show()

def test_model(model_path=None, n_components=50):
    """
    Purpose: Test a saved model on the test set and visualize results.
    Input:
        - model_path (str, optional): Path to the saved model.
        - n_components (int): Number of PCA components.
    Output: Tuple (accuracy, predictions, y_test) - Test accuracy, predictions, and true labels.
    """
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(n_components=n_components)
    
    # Load model
    model = load_model(model_path) if model_path else get_latest_model()
    
    # Fit scaler and PCA on training data
    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=n_components).fit(X_train)
    
    # Preprocess test data
    X_test_processed, _, _ = preprocess_data(X_test, scaler=scaler, pca=pca)
    
    # Evaluate on test set
    accuracy, predictions = evaluate_model(model, X_test_processed, y_test, dataset_name="Test")
    
    # Visualize results
    visualize_confusion_matrix(
        y_test,
        predictions,
        model_name="SVM with PCA",
        save_path=os.path.join(VISUALIZATION_DIR, "svm_pca_confusion_matrix.png")
    )
    
    # Visualize PCA variance
    visualize_pca_variance(
        pca,
        save_path=os.path.join(VISUALIZATION_DIR, "pca_variance_plot.png")
    )
    
    return accuracy, predictions, y_test