import os
import joblib
import torch
import numpy as np
from conditional_random_field import ConditionalRandomField
from sklearn_crfsuite import metrics, CRF

class ModelPersistence:
    def __init__(self, base_path="trained/"):
        """Initialize persistence with base path for model storage"""
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def load_model(self):
        """Load and return the requested models"""
        try:    
            model_path = os.path.join(self.base_path, 'crf_model.crfsuite')
            model = CRF(model_filename=model_path)
            return model
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return None