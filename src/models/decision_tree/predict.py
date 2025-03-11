import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.preprocess.decision_tree import load_data, compute_mean_std
from persistence import ModelPersistence

class PneumoniaPredictor:
    """Class for making predictions using trained model"""

    def __init__(self, model_path=None):
        """Initialize predictor with model"""
        self.persistence = ModelPersistence()
        if model_path:
            self.model = self.persistence.load_model(model_path)
        else:
            self.model = self.persistence.get_latest_model()
        
        # Initialize scaler
        self.scaler = StandardScaler()

    def preprocess_data(self, X):
        """Preprocess input data"""
        # Flatten the images if they're not already flattened
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled

    def predict(self, X):
        """Make predictions on preprocessed data"""
        X_processed = self.preprocess_data(X)
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)
        return predictions, probabilities

    def predict_single(self, image):
        """Make prediction for a single image"""
        if len(image.shape) == 2:
            image = image.reshape(1, -1)
        predictions, probabilities = self.predict(image)
        return predictions[0], probabilities[0]