import os 
import joblib
import pickle
from datetime import datetime

class ModelPersistence:
    """Class to handle Bayesian model persistence operations"""

    def __init__(self, model_dir="models/trained/bayesian"):
        """Initialize with model directory path"""
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def save_model(self, model, model_type, model_name=None):
        """Save Bayesian model to disk"""
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_type}_{timestamp}.joblib"
        
        model_path = os.path.join(self.model_dir, model_name)
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")
        return model_path

    def load_model(self, model_path):
        """Load model from disk"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        return model

    def get_latest_model(self, model_type=None):
        """Get the most recently saved model of specified type"""
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.joblib')]
        if model_type:
            model_files = [f for f in model_files if f.startswith(model_type)]
        
        if not model_files:
            raise FileNotFoundError("No saved models found")
        
        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(self.model_dir, x)))
        return self.load_model(os.path.join(self.model_dir, latest_model))