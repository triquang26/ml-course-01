import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# from src.models.conditional.persistence import ModelPersistence
from src.models.conditional_random_field.persistance import ModelPersistence
from conditional_random_field import ConditionalRandomField

class ModelPredictor:
    """Class for making predictions using trained genetic algorithm ensemble models"""
    
    def __init__(self, model: ConditionalRandomField):
        """Initialize predictor with model persistence"""
        self.model = model
    

    def predict(self, X):
        """Make predictions using the specified model"""
        try:
            y_pred = self.model.crf.predict(X)
            return y_pred
        except Exception as e:
            raise RuntimeError(f"Prediction error: {e}")