import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.preprocess.bayesian import load_data
from persistence import ModelPersistence

class BayesianPredictor:
    """Class for making predictions using trained Bayesian models"""

    def __init__(self, model_path=None, model_type=None):
        """Initialize predictor with model"""
        self.persistence = ModelPersistence()
        if model_path:
            self.model = self.persistence.load_model(model_path)
        else:
            self.model = self.persistence.get_latest_model(model_type)
        
        self.scaler = StandardScaler()

    def preprocess_data(self, X, model_type='gaussian'):
        """Preprocess input data based on model type"""
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        if model_type == 'gaussian':
            # Scale features for Gaussian NB
            X_processed = self.scaler.fit_transform(X)
        elif model_type == 'bernoulli':
            # Binarize features for Bernoulli NB
            X_processed = (X > 0.5).astype(np.int32)
        else:
            # Use raw counts for Multinomial NB
            X_processed = X.astype(np.int32)
            
        return X_processed

    def predict(self, X, model_type='gaussian'):
        """Make predictions on preprocessed data"""
        X_processed = self.preprocess_data(X, model_type)
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)
        return predictions, probabilities

    def predict_single(self, image, model_type='gaussian'):
        """Make prediction for a single image"""
        if len(image.shape) == 2:
            image = image.reshape(1, -1)
        predictions, probabilities = self.predict(image, model_type)
        return predictions[0], probabilities[0]