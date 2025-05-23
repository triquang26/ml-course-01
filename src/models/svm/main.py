import os
import sys
import time
from typing import Dict, Tuple, List
import numpy as np

# Add project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from medmnist import PneumoniaMNIST

# Import from project modules
from src.data.preprocess.svm import load_data

from src.visualization.visualize import (
    visualize_results,
    compare_models,
    VISUALIZATION_DIR
)

from src.models.svm.core.svm import run_svm, visualize_results as visualize_svm_results
from src.models.svm.persistence import ModelPersistence
from model_tuning import ModelTuner

class ModelParameters:
    def __init__(self,
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 gamma: str = 'scale',
                 tune: bool = False,
                 tuning_method: str = 'grid',
                 save_models: bool = True):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.tune = tune
        self.tuning_method = tuning_method
        self.save_models = save_models

class ModelRunner:
    def __init__(self, params: ModelParameters):
        self.params = params
        self.results = {}
        self.tuning_results = {}
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.ensure_dir_exists(VISUALIZATION_DIR)
        self.persistence = ModelPersistence()
        
    @staticmethod
    def ensure_dir_exists(directory_path):
        """Ensure a directory exists, creating it if necessary"""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Created directory: {directory_path}")
            
    def tune_model(self, model_name: str, train_dataset):
        """Tune model parameters using cross-validation"""
        X_train = train_dataset.imgs.astype('float32') / 255.0
        X_train = X_train.reshape(X_train.shape[0], -1)
        y_train = train_dataset.labels.flatten()
        
        param_grids = {
            "Support Vector Machine": {
                'kernel': ['linear', 'rbf', 'poly'],
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
        }
        
        if model_name in param_grids:
            tuner = ModelTuner(model_name, param_grids[model_name])
            best_params, best_score = tuner.tune_parameters(
                X_train, y_train, 
                method=self.params.tuning_method
            )
            
            # Save tuning visualization
            tuner.plot_tuning_results(
                save_path=os.path.join(VISUALIZATION_DIR,
                                     f"{self.timestamp}_{model_name}_tuning.png")
            )
            
            self.tuning_results[model_name] = {
                'best_params': best_params,
                'best_score': best_score
            }
            return best_params
        return {}

    def run_model(self, model_name: str, model_func, train_dataset, test_dataset, **kwargs):
        """Run SVM model with optional parameter tuning"""
        if self.params.tune:
            best_params = self.tune_model(model_name, train_dataset)
            kwargs.update(best_params)
        
        # Ensure parameters are correctly passed
        base_params = {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
        }
        
        # Add default parameters from self.params
        for param in ['kernel', 'C', 'gamma']:
            if hasattr(self.params, param) and param not in kwargs:
                base_params[param] = getattr(self.params, param)
        
        print(f"\nRunning {model_name} with parameters:")
        all_params = {**base_params, **kwargs}
        for key, value in all_params.items():
            if key not in ['train_dataset', 'test_dataset']:
                print(f"  {key}: {value}")
        
        # Run model and get results
        model_result = model_func(**all_params)
        
        accuracy, predictions, actual_labels, model, preprocessors = model_result
        
        # Save model if enabled
        if self.params.save_models:
            # Clean model name for persistence
            clean_model_name = model_name.lower().replace(' ', '_')
            self.persistence.save_model(model, clean_model_name, preprocessors)
        
        # Visualize and save results
        # First use the SVM's visualization function
        visualize_svm_results(
            predictions=predictions, 
            actual_labels=actual_labels,
            save_path=os.path.join(VISUALIZATION_DIR, 
                                f"{self.timestamp}_{model_name.lower().replace(' ', '_')}_cm.png")
        )
        
        # Then use project's standard visualization for metrics
        metrics = visualize_results(
            y_true=actual_labels,
            y_pred=predictions,
            model_name=model_name,
            save_path=os.path.join(VISUALIZATION_DIR, 
                                f"{self.timestamp}_{model_name.lower().replace(' ', '_')}.png")
        )
        
        self.results[model_name] = metrics
        return metrics

    def run_all_models(self):
        """Run SVM model"""
        train_dataset, test_dataset = load_data()
        
        # Run Support Vector Machine
        self.run_model("Support Vector Machine",
                      run_svm,
                      train_dataset,
                      test_dataset)

    def print_summary(self):
        """Print and visualize model comparison results"""
        print("\n" + "="*80)
        print("MODEL SUMMARY")
        print("="*80)
        
        for model_name, metrics in self.results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}") 
            print("-" * 40)

def main():
    """Main execution function"""
    params = ModelParameters(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        tune=True,
        tuning_method='grid',
        save_models=True
    )
    
    runner = ModelRunner(params)
    runner.run_all_models()
    runner.print_summary()

if __name__ == "__main__":
    main()