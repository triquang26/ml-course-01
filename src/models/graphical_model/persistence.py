import os
import pickle
from medmnist import PneumoniaMNIST
import os

MODEL_DIR = "trained"

class ModelPersistence:
    @staticmethod
    def load_data(split='train'):
        """Load dataset split (train/test)"""
        try:
            dataset = PneumoniaMNIST(split=split, download=True)
            print(f"Loaded {split} dataset: {len(dataset)} samples")
            return dataset
        except Exception as e:
            print(f"Error loading {split} data: {e}")
            return None
        
    @staticmethod
    def save_model(model, model_name: str):
        """Save trained model"""
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f"{model_name.lower().replace(' ', '_')}.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to: {model_path}")
            
    @staticmethod
    def load_model(model_name: str):
        """Load saved model"""
        model_path = os.path.join(MODEL_DIR, f"{model_name.lower().replace(' ', '_')}.pkl")
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except FileNotFoundError:
            print(f"No saved model found at: {model_path}")
            return None