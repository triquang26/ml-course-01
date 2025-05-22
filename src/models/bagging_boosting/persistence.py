import os
import joblib
from datetime import datetime

class ModelPersistence:
    """Class to handle Bagging and Boosting model persistence operations"""

    def __init__(self, model_dir="models/trained/bagging_boosting"):
        """Initialize with model directory path"""
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def save_model(self, model, model_type, model_name=None):
        """
        Save model to disk using joblib (.joblib extension).
        
        Args:
            model: the model object to save
            model_type: a short string identifying the model (used as filename prefix)
            model_name: optional full filename; if None, auto-generates one
        Returns:
            The path to the saved model file.
        """
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_type}_{timestamp}.joblib"

        model_path = os.path.join(self.model_dir, model_name)
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")
        return model_path

    def load_model(self, model_path):
        """
        Load a joblib-saved model from disk.
        
        Args:
            model_path: path to the .joblib file
        Returns:
            The loaded model object.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        return model

    def get_latest_model(self, model_type=None):
        """
        Find and load the most recently saved .joblib model of a given type.
        
        Args:
            model_type: optional prefix to filter model files
        Returns:
            The loaded most-recent model.
        """
        # List only .joblib files
        model_files = [
            f for f in os.listdir(self.model_dir)
            if f.endswith(".joblib")
        ]
        if model_type:
            model_files = [f for f in model_files if f.startswith(model_type)]

        if not model_files:
            raise FileNotFoundError("No saved models found")

        # Pick the file with the latest creation time
        latest_model = max(
            model_files,
            key=lambda fn: os.path.getctime(os.path.join(self.model_dir, fn))
        )
        return self.load_model(os.path.join(self.model_dir, latest_model))
