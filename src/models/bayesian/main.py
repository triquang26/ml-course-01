import os
import time
import sys
from medmnist import INFO, PneumoniaMNIST
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
# Import the three Naive Bayes model implementations
from src.models.bayesian.core.gaussian_naive_bayes import run_gaussian_naive_bayes, visualize_results as gnb_visualize
from src.models.bayesian.core.bernoulli_naive_bayes import run_bernoulli_naive_bayes, visualize_results as bnb_visualize
from src.models.bayesian.core.multinomial_naive_bayes import run_multinomial_naive_bayes, visualize_results as mnb_visualize
from persistence import ModelPersistence
from src.data.preprocess.bayesian import (
    load_data,
    ensure_dir_exists
)

from src.visualization.visualize import (
    visualize_results,
    compare_models,
    VISUALIZATION_DIR
)

def main():
    """Main function to run all Naive Bayes models"""
    # Create visualization directory if it doesn't exist
    ensure_dir_exists(VISUALIZATION_DIR)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Initialize model persistence
    model_persistence = ModelPersistence()
    
    # Define which models to run
    run_gnb = True  # Gaussian Naive Bayes
    run_bnb = True  # Bernoulli Naive Bayes
    run_mnb = True  # Multinomial Naive Bayes
    
    # Load data
    train_dataset, test_dataset = load_data()
    
    # Track results for comparison
    results = {}
    
    # Run Gaussian Naive Bayes
    if run_gnb:
        gnb_accuracy, gnb_predictions, gnb_actual_labels, gnb_model = run_gaussian_naive_bayes(
            train_dataset, 
            test_dataset
        )
        # Save the Gaussian NB model
        model_path = model_persistence.save_model(
            gnb_model, 
            'gaussian_nb', 
            f"gaussian_nb_{timestamp}.joblib"
        )
        
        gnb_visualize(
            gnb_predictions, 
            gnb_actual_labels,
            save_path=os.path.join(VISUALIZATION_DIR, f"{timestamp}_gaussian_nb_cm.png")
        )
        results['Gaussian NB'] = gnb_accuracy
    
    # Run Bernoulli Naive Bayes
    if run_bnb:
        bnb_accuracy, bnb_predictions, bnb_actual_labels, bnb_model = run_bernoulli_naive_bayes(
            train_dataset, 
            test_dataset
        )
        # Save the Bernoulli NB model
        model_path = model_persistence.save_model(
            bnb_model, 
            'bernoulli_nb', 
            f"bernoulli_nb_{timestamp}.joblib"
        )
        
        bnb_visualize(
            bnb_predictions, 
            bnb_actual_labels,
            save_path=os.path.join(VISUALIZATION_DIR, f"{timestamp}_bernoulli_nb_cm.png")
        )
        results['Bernoulli NB'] = bnb_accuracy
    
    # Run Multinomial Naive Bayes
    if run_mnb:
        mnb_accuracy, mnb_predictions, mnb_actual_labels, mnb_model = run_multinomial_naive_bayes(
            train_dataset, 
            test_dataset
        )
        # Save the Multinomial NB model
        model_path = model_persistence.save_model(
            mnb_model, 
            'multinomial_nb', 
            f"multinomial_nb_{timestamp}.joblib"
        )
        
        mnb_visualize(
            mnb_predictions, 
            mnb_actual_labels,
            save_path=os.path.join(VISUALIZATION_DIR, f"{timestamp}_multinomial_nb_cm.png")
        )
        results['Multinomial NB'] = mnb_accuracy

    # Display comparative results
    if results:
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        for model_name, accuracy in results.items():
            print(f"{model_name}: Accuracy = {accuracy:.4f}")

if __name__ == "__main__":
    main()