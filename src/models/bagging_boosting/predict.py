import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler

# make sure project root is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.preprocess.bagging_boosting import load_data
from persistence import ModelPersistence

class BaggingBoostingPredictor:
    """Load a trained Bagging/Boosting model and make predictions on new data."""

    def __init__(self, model_path=None, model_type=None):
        """
        model_path: explicit path to a .joblib model
        model_type: prefix to find the latest model saved by ModelPersistence
        """
        self.persistence = ModelPersistence()

        if model_path:
            self.model = self.persistence.load_model(model_path)
        else:
            self.model = self.persistence.get_latest_model(model_type)

        # if it supports predict_proba, assume it's sklearn and we should standardize
        if hasattr(self.model, 'predict_proba'):
            self.is_sklearn = True
            self.scaler = StandardScaler()
        else:
            self.is_sklearn = False

    def preprocess_data(self, X):
        """
        Flatten images and normalize to [0,1].
        Expects X of shape (n_samples, 28, 28) or (n_samples, n_features).
        """
        # flatten
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        # normalize
        X = X.astype(np.float32) / 255.0
        return X

    def predict(self, X):
        """
        X: array of shape (n_samples, 28, 28) or (n_samples, n_features)
        Returns: (predictions, probabilities or None)
        """
        X_proc = self.preprocess_data(X)

        if self.is_sklearn:
            # scale then predict
            X_scaled = self.scaler.fit_transform(X_proc)
            y_pred   = self.model.predict(X_scaled)
            y_prob   = self.model.predict_proba(X_scaled)
        else:
            # scratch models expect normalized, flattened features
            y_pred = self.model.predict(X_proc)
            y_prob = None

        return y_pred, y_prob

    def predict_single(self, image):
        """
        image: single 28×28 array, returns (pred_label, pred_proba or None)
        """
        # flatten+batch dimension
        arr = image.reshape(1, -1) if image.ndim == 2 else image.reshape(1, -1)
        preds, probs = self.predict(arr)
        return preds[0], (probs[0] if probs is not None else None)
