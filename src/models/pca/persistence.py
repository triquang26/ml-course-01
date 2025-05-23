import os
import joblib
from datetime import datetime

# Define directory
MODEL_DIR = "/content/trained"

def ensure_dir_exists(directory):
    """
    Purpose: Create a directory if it does not exist.
    Input: directory (str) - Path to the directory.
    Output: None
    """
    os.makedirs(directory, exist_ok=True)

def save_model(model, model_dir=MODEL_DIR, model_name=None):
    """
    Purpose: Save a trained model to disk with a timestamped filename.
    Input:
        - model: Trained model object (e.g., SVM).
        - model_dir (str): Directory to save the model.
        - model_name (str, optional): Custom name for the model file.
    Output: model_path (str) - Path where the model is saved.
    """
    ensure_dir_exists(model_dir)
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"svm_pca_{timestamp}.joblib"
    model_path = os.path.join(model_dir, model_name)
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    return model_path

def load_model(model_path):
    """
    Purpose: Load a saved model from disk.
    Input: model_path (str) - Path to the model file.
    Output: model - Loaded model object.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    return model

def get_latest_model(model_dir=MODEL_DIR):
    """
    Purpose: Retrieve the most recently saved model from the model directory.
    Input: model_dir (str) - Directory containing model files.
    Output: model - Loaded model object.
    """
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    if not model_files:
        raise FileNotFoundError("No saved models found")
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    return load_model(os.path.join(model_dir, latest_model))

