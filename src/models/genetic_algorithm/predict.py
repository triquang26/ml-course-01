import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.models.genetic_algorithm.persistence import ModelPersistence

class ModelPredictor:
    """Class for making predictions using trained genetic algorithm ensemble models"""
    
    def __init__(self):
        """Initialize predictor with model persistence"""
        self.persistence = ModelPersistence()
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_model(self, model_name: str):
        """Load a specific model from storage"""
        try:
            model = self.persistence.load_model(model_name)
            self.models[model_name] = model
            return model
        except Exception as e:
            raise ValueError(f"Error loading model {model_name}: {e}")

    def predict(self, X: np.ndarray, model_name: str) -> np.ndarray:
        """Make predictions using the specified model"""
        try:
            # Basic preprocessing
            if len(X.shape) > 2:
                X = X.reshape(X.shape[0], -1)  # Flatten images
            X = X.astype('float32') / 255.0  # Normalize
            
            # Get model
            model = self.models.get(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not found")

            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Make predictions using ensemble
            predictions = self._predict_ensemble(X_scaled, model)
            
            return predictions
            
        except Exception as e:
            raise RuntimeError(f"Prediction error: {e}")

    def _predict_ensemble(self, X: np.ndarray, model) -> np.ndarray:
        """Predict using the ensemble of models"""
        predictions = []
        
        for sample in X:
            # Get predictions from each model in the ensemble
            sample = sample.reshape(1, -1)
            model_predictions = []
            
            # Get predictions from CNN
            cnn_pred = model['cnn'].predict(sample.reshape(-1, 28, 28, 1))[0]
            model_predictions.append(cnn_pred)
            
            # Get predictions from Decision Tree
            dt_pred = model['decision_tree'].predict(sample)[0]
            model_predictions.append(dt_pred)
            
            # Get predictions from Naive Bayes
            nb_pred = model['naive_bayes'].predict(sample)[0]
            model_predictions.append(nb_pred)
            
            # Combine predictions using weighted voting
            weights = model['weights']
            weighted_pred = np.average(model_predictions, weights=weights)
            final_pred = 1 if weighted_pred >= 0.5 else 0
            
            predictions.append(final_pred)
                
        return np.array(predictions)

    def predict_single(self, image: np.ndarray, model_name: str) -> int:
        """Make prediction for a single image"""
        if len(image.shape) == 2:
            image = image.reshape(1, -1)
        predictions = self.predict(image, model_name)
        return predictions[0]