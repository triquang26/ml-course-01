import os
import time
import argparse
import importlib.util
import sys
from contextlib import contextmanager

@contextmanager
def add_to_path(p):
    """Temporarily add a directory to sys.path"""
    old_path = sys.path.copy()
    sys.path.insert(0, p)
    try:
        yield
    finally:
        sys.path = old_path

def import_main_from_path(module_path):
    """Dynamically import a main function from a given file path"""
    try:
        # Get the directory containing the module
        module_dir = os.path.dirname(module_path)
        module_name = os.path.basename(module_dir)
        
        # Temporarily add the module's directory to sys.path so it can find its own imports
        with add_to_path(module_dir):
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.main
    except Exception as e:
        print(f"Failed to import {module_path}: {str(e)}")
        return None

# Define paths to main.py files
BAYESIAN_MAIN = os.path.join(os.path.dirname(__file__), 'bayesian', 'main.py')
DECISION_TREE_MAIN = os.path.join(os.path.dirname(__file__), 'decision_tree', 'main.py')
GENETIC_ALGORITHM_MAIN = os.path.join(os.path.dirname(__file__), 'genetic_algorithm', 'main.py')
GRAPHICAL_MODEL_MAIN = os.path.join(os.path.dirname(__file__), 'graphical_model', 'main.py')
NEURAL_NETWORK_MAIN = os.path.join(os.path.dirname(__file__), 'neural_network', 'main.py')
BAGGING_BOOSTING_MAIN = os.path.join(os.path.dirname(__file__), 'bagging_boosting', 'main.py')
# Import main functions dynamically
bayesian_main = import_main_from_path(BAYESIAN_MAIN)
decision_tree_main = import_main_from_path(DECISION_TREE_MAIN)
genetic_algorithm_main = import_main_from_path(GENETIC_ALGORITHM_MAIN)
graphical_model_main = import_main_from_path(GRAPHICAL_MODEL_MAIN)
neural_network_main = import_main_from_path(NEURAL_NETWORK_MAIN)
bagging_boosting_main = import_main_from_path(BAGGING_BOOSTING_MAIN)
def run_model_safely(model_name, model_func):
    """Run a model function with proper error handling"""
    if model_func is None:
        print(f"⚠️ {model_name} model was not loaded properly, skipping...")
        return False
    
    try:
        # Add the model's directory to sys.path temporarily
        model_dir = os.path.dirname(globals()[f"{model_name.upper()}_MAIN"])
        with add_to_path(model_dir):
            model_func()
        return True
    except Exception as e:
        print(f"❌ Error running {model_name} model: {str(e)}")
        return False

def train_all_models():
    """Train all models sequentially"""
    start_time = time.time()
    successful_models = 0
    
    print("\n" + "="*80)
    print("TRAINING PNEUMONIAMNIST CLASSIFICATION MODELS")
    print("="*80)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports/figures")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Ensuring results directory exists: {results_dir}")
    
    # Train Bayesian models
    print("\n" + "="*80)
    print("RUNNING BAYESIAN MODELS")
    print("="*80)
    if run_model_safely("bayesian", bayesian_main):
        successful_models += 1
    
    # Train Decision Tree model
    print("\n" + "="*80)
    print("RUNNING DECISION TREE MODELS")
    print("="*80)
    if run_model_safely("decision_tree", decision_tree_main):
        successful_models += 1
    
    # Train Genetic Algorithm ensemble
    print("\n" + "="*80)
    print("RUNNING GENETIC ALGORITHM ENSEMBLE")
    print("="*80)
    if run_model_safely("genetic_algorithm", genetic_algorithm_main):
        successful_models += 1
    
    # Train Graphical models
    print("\n" + "="*80)
    print("RUNNING GRAPHICAL MODELS")
    print("="*80)
    if run_model_safely("graphical_model", graphical_model_main):
        successful_models += 1
    
    # Train Neural Network models
    print("\n" + "="*80)
    print("RUNNING NEURAL NETWORK MODELS")
    print("="*80)
    if run_model_safely("neural_network", neural_network_main):
        successful_models += 1
    
    # Train Bagging and Boosting models
    print("\n" + "="*80)
    print("RUNNING BAGGING AND BOOSTING MODELS")
    print("="*80)
    if run_model_safely("bagging_boosting", bagging_boosting_main):
        successful_models += 1
    # Print execution time
    execution_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"TRAINING COMPLETE: {successful_models} of 6 models trained successfully")
    print(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print("="*80)


def train_selected_models(models):
    """Train only selected models"""
    start_time = time.time()
    successful_models = 0
    total_models = len(models)
    
    print("\n" + "="*80)
    print("TRAINING SELECTED PNEUMONIAMNIST CLASSIFICATION MODELS")
    print("="*80)
    print(f"Models selected: {', '.join(models)}")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports/figures")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Ensuring results directory exists: {results_dir}")
    
    # Train selected models
    if "bayesian" in models:
        print("\n" + "="*80)
        print("RUNNING BAYESIAN MODELS")
        print("="*80)
        if run_model_safely("bayesian", bayesian_main):
            successful_models += 1
    
    if "decision_tree" in models:
        print("\n" + "="*80)
        print("RUNNING DECISION TREE MODELS")
        print("="*80)
        if run_model_safely("decision_tree", decision_tree_main):
            successful_models += 1
    
    if "genetic_algorithm" in models:
        print("\n" + "="*80)
        print("RUNNING GENETIC ALGORITHM ENSEMBLE")
        print("="*80)
        if run_model_safely("genetic_algorithm", genetic_algorithm_main):
            successful_models += 1
    
    if "graphical_model" in models:
        print("\n" + "="*80)
        print("RUNNING GRAPHICAL MODELS")
        print("="*80)
        if run_model_safely("graphical_model", graphical_model_main):
            successful_models += 1
    
    if "neural_network" in models:
        print("\n" + "="*80)
        print("RUNNING NEURAL NETWORK MODELS")
        print("="*80)
        if run_model_safely("neural_network", neural_network_main):
            successful_models += 1

    if "bagging_boosting" in models:
        print("\n" + "="*80)
        print("RUNNING BAGGING AND BOOSTING MODELS")
        print("="*80)
        if run_model_safely("bagging_boosting", bagging_boosting_main):
            successful_models += 1
    # Print execution time
    execution_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"TRAINING COMPLETE: {successful_models} of {total_models} models trained successfully")
    print(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train machine learning models for PneumoniaMNIST classification")
    parser.add_argument("--models", type=str, nargs="+", 
                        choices=["bayesian", "decision_tree", "genetic_algorithm", "graphical_model", "neural_network", "bagging_boosting", "all"],
                        default=["all"], 
                        help="Specify which models to train ('all' to train all models)")
    
    args = parser.parse_args()
    
    if "all" in args.models or len(args.models) == 0:
        train_all_models()
    else:
        train_selected_models(args.models)