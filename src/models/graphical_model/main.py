import os
import sys
import time
from typing import Dict, Tuple, List
import numpy as np

# Add project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from medmnist import PneumoniaMNIST

# Import from project modules
from src.data.preprocess.graphical_model import (
    load_data,
    extract_data_labels,
    discretize_features,
    ensure_dir_exists
)

from src.features.features_selection.graphical_model import (
    apply_pca,
    calculate_feature_correlation
)

from src.visualization.visualize import (
    visualize_results,
    compare_models,
    VISUALIZATION_DIR
)

from src.models.graphical_model.core.bayesian_network_graphical import run_bayesian_network
from src.models.graphical_model.core.augmentated_naive_bayes_graphical import run_augmented_naive_bayes
from src.models.graphical_model.core.hidden_markov_graphical import run_hidden_markov_model
from src.models.graphical_model.persistence import ModelPersistence
from model_tuning import ModelTuner

class ModelParameters:
    def __init__(self,
                 n_components: int = 50,
                 n_bins: int = 10,
                 n_hidden_states: int = 8,
                 correlation_threshold: float = 0.5,
                 tune: bool = False,
                 tuning_method: str = 'grid',
                 save_models: bool = True):
        self.n_components = n_components
        self.n_bins = n_bins
        self.n_hidden_states = n_hidden_states
        self.correlation_threshold = correlation_threshold
        self.tune = tune
        self.tuning_method = tuning_method
        self.save_models = save_models
class ModelRunner:
    def __init__(self, params: ModelParameters):
        self.params = params
        self.results = {}
        self.tuning_results = {}
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        ensure_dir_exists(VISUALIZATION_DIR)
        self.persistence = ModelPersistence()
        
    def tune_model(self, model_name: str, train_dataset):
        X_train, y_train = extract_data_labels(train_dataset)
        
        param_grids = {
            "Bayesian Network": {
                'n_components': [30, 50, 70],
                'n_bins': [5, 10, 15]
            },
            "Augmented Naive Bayes": {
                'correlation_threshold': [0.3, 0.5, 0.7],
                'n_components': [30, 50, 70]
            },
            "Hidden Markov Model": {
                'n_hidden_states': [4, 8, 12],
                'n_components': [30, 50, 70]
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
        """Run a specific graphical model with optional parameter tuning"""
        if self.params.tune:
            best_params = self.tune_model(model_name, train_dataset)
            kwargs.update(best_params)
        
        # Đảm bảo tham số được truyền đúng
        base_params = {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
        }
        
        # Thêm tham số mặc định từ self.params
        for param in ['n_components', 'n_bins', 'n_hidden_states', 'correlation_threshold']:
            if hasattr(self.params, param) and param not in kwargs:
                base_params[param] = getattr(self.params, param)
        
        print(f"\nChạy {model_name} với tham số:")
        all_params = {**base_params, **kwargs}
        for key, value in all_params.items():
            if key not in ['train_dataset', 'test_dataset']:
                print(f"  {key}: {value}")
        
        # Chạy mô hình và lấy kết quả
        model_result = model_func(**all_params)
        
        # Check if the model function returns preprocessors or not
        if len(model_result) >= 5:  # Updated this check 
            accuracy, predictions, actual_labels, model, preprocessors = model_result
        else:
            accuracy, predictions, actual_labels, model = model_result
            preprocessors = None  # Set preprocessors to None if not returned
        
        # Save model if enabled
        if self.params.save_models:
            # Clean model name for persistence
            clean_model_name = model_name.lower().replace(' ', '_')
            self.persistence.save_model(model, clean_model_name, preprocessors)
        
        # Visualize and save results
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
        """Run all graphical models"""
        train_dataset, test_dataset = load_data()
        
        # Run Bayesian Network
        self.run_model("Bayesian Network",
                      run_bayesian_network,
                      train_dataset,
                      test_dataset)
                      
        # Run Augmented Naive Bayes
        self.run_model("Augmented Naive Bayes",
                      run_augmented_naive_bayes, 
                      train_dataset,
                      test_dataset,
                      correlation_threshold=self.params.correlation_threshold)
                      
        # Run Hidden Markov Model  
        self.run_model("Hidden Markov Model",
                      run_hidden_markov_model,
                      train_dataset, 
                      test_dataset,
                      n_hidden_states=self.params.n_hidden_states)

    def print_summary(self):
        """Print and visualize model comparison results"""
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        for model_name, metrics in self.results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}") 
            print("-" * 40)
            
        if self.results:
            compare_models(
                self.results,
                save_path=os.path.join(VISUALIZATION_DIR,
                                     f"{self.timestamp}_model_comparison.png")
            )

def main():
    """Main execution function"""
    params = ModelParameters(
        n_components=50,
        n_bins=10,
        n_hidden_states=8,
        correlation_threshold=0.5,
        tune=True,
        tuning_method='grid',
        save_models=True  # Enable model persistence
    )
    
    runner = ModelRunner(params)
    runner.run_all_models()
    runner.print_summary()

if __name__ == "__main__":
    main()