import os
import joblib
import torch
import numpy as np

class ModelPersistence:
    def __init__(self, base_path="trained/"):
        """Initialize persistence with base path for model storage"""
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save_models(self, dt_model=None, bayes_model=None, cnn_model=None, ga_weights=None):
        """Save all models to their respective files"""
        try:
            if dt_model is not None:
                joblib.dump(dt_model, os.path.join(self.base_path, "trained_decision_tree.joblib"))
            
            if bayes_model is not None:
                joblib.dump(bayes_model, os.path.join(self.base_path, "trained_naive_bayes.joblib"))
            
            if cnn_model is not None:
                torch.save(cnn_model.state_dict(), os.path.join(self.base_path, "trained_cnn_model.pth"))
            
            if ga_weights is not None:
                np.save(os.path.join(self.base_path, "ga_best_weights.npy"), ga_weights)
            
            print("Models saved successfully")
            return True
        except Exception as e:
            print(f"Error saving models: {str(e)}")
            return False

    def load_models(self, load_dt=True, load_bayes=True, load_cnn=True, load_ga=True):
        """Load and return the requested models"""
        models = {}
        try:
            if load_dt:
                dt_path = os.path.join(self.base_path, "trained_decision_tree.joblib")
                if os.path.exists(dt_path):
                    models['decision_tree'] = joblib.load(dt_path)
            
            if load_bayes:
                bayes_path = os.path.join(self.base_path, "trained_naive_bayes.joblib")
                if os.path.exists(bayes_path):
                    models['naive_bayes'] = joblib.load(bayes_path)
            
            if load_cnn:
                cnn_path = os.path.join(self.base_path, "trained_cnn_model.pth")
                if os.path.exists(cnn_path):
                    models['cnn_weights'] = torch.load(cnn_path)
            
            if load_ga:
                ga_path = os.path.join(self.base_path, "ga_best_weights.npy")
                if os.path.exists(ga_path):
                    models['ga_weights'] = np.load(ga_path)
            
            return models
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return {}