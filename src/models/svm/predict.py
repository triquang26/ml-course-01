import os
import sys
import numpy as np
import pickle
import logging

from .persistence import ModelPersistence

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.data.preprocess.svm import load_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelPredictor:
    """
    Class for loading trained SVM models and making predictions
    """
    def __init__(self):
        self.model = None
        self.model_name = None
        self.params = {}
        self.persistence = ModelPersistence()
        
    def load_model(self, model_name):
        """
        Load a trained model and its parameters from disk
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Load the model
            self.model = self.persistence.load_model(model_name)
            if self.model is None:
                return False
                
            # Load preprocessors (parameters)
            preprocessors = self.persistence.load_preprocessors(model_name)
            self.params = preprocessors.get('params') or {}
            
            self.model_name = model_name
            return True
            
        except Exception as e:
            logger.error(f"Error loading {model_name}: {str(e)}")
            return False
            
    def preprocess_data(self, X):
        """
        Apply appropriate preprocessing to input data
        
        Args:
            X: Input features to preprocess
            
        Returns:
            Preprocessed data
        """
        logger.info("Preprocessing data for SVM prediction...")
        
        # Create a copy to avoid modifying the original data
        X_processed = X.copy()
        
        # For SVM, we just need to flatten the images and normalize
        if len(X_processed.shape) > 2:
            # If input is image data, flatten it
            X_processed = X_processed.reshape(X_processed.shape[0], -1)
            
        # Normalize data if needed
        if X_processed.max() > 1.0:
            X_processed = X_processed.astype('float32') / 255.0
            
        logger.info(f"Preprocessed data shape: {X_processed.shape}")
        return X_processed
        
    def predict(self, X):
        """
        Make predictions using the loaded model
        
        Args:
            X: Input features to predict on
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")
                
        # Preprocess the data
        X_processed = self.preprocess_data(X)
        
        # Make predictions
        try:
            logger.info("Making predictions with SVM model...")
            predictions = self.model.predict(X_processed)
            
            # Log prediction statistics
            class_counts = np.bincount(predictions.astype(int), minlength=2)
            logger.info(f"Prediction distribution: {class_counts}")
            
            return predictions
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            # Return balanced predictions as fallback
            n_samples = X_processed.shape[0]
            return np.random.choice([0, 1], size=n_samples)

def predict_with_svm(model, data):
    """
    Make predictions using the SVM model
    
    Args:
        model: SVM model
        data: Input data
        
    Returns:
        Predictions array
    """
    # Preprocess data by flattening and normalizing
    if len(data.shape) > 2:
        data_flat = data.reshape(data.shape[0], -1)
    else:
        data_flat = data
        
    if data_flat.max() > 1.0:
        data_flat = data_flat.astype('float32') / 255.0
    
    # Make predictions
    try:
        predictions = model.predict(data_flat)
        return predictions
    except Exception as e:
        logging.error(f"Error during SVM prediction: {str(e)}")
        return np.zeros(len(data_flat))