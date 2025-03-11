import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import optuna
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from typing import Dict, Any, List, Iterator

class ModelTuner:
    def __init__(self, model_name: str, param_grid: Dict[str, Any], cv: int = 5):
        self.model_name = model_name
        self.param_grid = param_grid
        self.cv = cv
        self.best_params = None  
        self.best_score = None
        self.tuning_history = []

    def _parameter_combinations(self) -> Iterator[Dict[str, Any]]:
        """Generate all possible parameter combinations from param_grid"""
        param_names = sorted(self.param_grid)
        param_values = [self.param_grid[name] for name in param_names]
        
        for values in product(*param_values):
            yield dict(zip(param_names, values))

    def _evaluate_params(self, X, y, params: Dict[str, Any]) -> float:
        """Evaluate a parameter combination using cross-validation"""
        # Here you would normally instantiate your model with params and evaluate
        # For demonstration, return random score between 0-1
        scores = np.random.uniform(0, 1, size=self.cv)
        return scores.mean()

    def _grid_search(self, X, y):
        """Perform grid search over parameter combinations"""
        results = []
        best_score = -np.inf
        best_params = None
        
        # Try each parameter combination
        for params in self._parameter_combinations():
            score = self._evaluate_params(X, y, params)
            results.append({'params': params, 'score': score})
            
            if score > best_score:
                best_score = score
                best_params = params
                
        self.best_params = best_params
        self.best_score = best_score
        self.tuning_history = results
        
        return best_params, best_score

    def tune_parameters(self, X, y, method='grid', n_trials=100):
        if method == 'grid':
            return self._grid_search(X, y)
        elif method == 'random':
            return self._random_search(X, y, n_trials)
        elif method == 'optuna':
            return self._optuna_optimize(X, y, n_trials)
        else:
            raise ValueError(f"Unknown tuning method: {method}")

    def plot_tuning_results(self, save_path=None):
        """Plot the tuning history"""
        if not self.tuning_history:
            print("No tuning history available")
            return
            
        scores = [result['score'] for result in self.tuning_history]
        plt.figure(figsize=(10, 6))
        plt.plot(scores)
        plt.title(f'Tuning History - {self.model_name}')
        plt.xlabel('Trial')
        plt.ylabel('Score')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()