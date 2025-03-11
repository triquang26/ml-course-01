import os
import time
from medmnist import INFO, PneumoniaMNIST
import matplotlib.pyplot as plt
import numpy as np

# Import the three Naive Bayes model implementations
from gaussian_naive_bayes import run_gaussian_naive_bayes, visualize_results as gnb_visualize
from bernoulli_naive_bayes import run_bernoulli_naive_bayes, visualize_results as bnb_visualize
from multinomial_naive_bayes import run_multinomial_naive_bayes, visualize_results as mnb_visualize

# Define visualization directory
VISUALIZATION_DIR = "reports/figures"

def load_data():
    """Load the PneumoniaMNIST dataset"""
    print("Loading PneumoniaMNIST dataset...")
    
    # Load dataset information
    dataset_info = INFO["pneumoniamnist"]
    print(f"Dataset description: {dataset_info['description']}")
    print(f"Number of classes: {len(dataset_info['label'])}, Labels: {dataset_info['label']}")
    
    # Load training and test sets
    train_dataset = PneumoniaMNIST(split='train', download=True)
    test_dataset = PneumoniaMNIST(split='test', download=True)
    
    print("Dataset loaded successfully.")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def main():
    """Main function to run all Naive Bayes models"""
    # Create visualization directory if it doesn't exist
    ensure_dir_exists(VISUALIZATION_DIR)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
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
        gnb_accuracy, gnb_predictions, gnb_actual_labels = run_gaussian_naive_bayes(
            train_dataset, 
            test_dataset
        )
        gnb_visualize(
            gnb_predictions, 
            gnb_actual_labels,
            save_path=os.path.join(VISUALIZATION_DIR, f"{timestamp}_gaussian_nb_cm.png")
        )
        results['Gaussian NB'] = gnb_accuracy
    
    # Run Bernoulli Naive Bayes
    if run_bnb:
        bnb_accuracy, bnb_predictions, bnb_actual_labels = run_bernoulli_naive_bayes(
            train_dataset, 
            test_dataset
        )
        bnb_visualize(
            bnb_predictions, 
            bnb_actual_labels,
            save_path=os.path.join(VISUALIZATION_DIR, f"{timestamp}_bernoulli_nb_cm.png")
        )
        results['Bernoulli NB'] = bnb_accuracy
    
    # Run Multinomial Naive Bayes
    if run_mnb:
        mnb_accuracy, mnb_predictions, mnb_actual_labels = run_multinomial_naive_bayes(
            train_dataset, 
            test_dataset
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
        
        # Plot comparative results
        plt.figure(figsize=(10, 6))
        plt.bar(results.keys(), results.values())
        plt.title('Model Comparison - Accuracy')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        for i, (model, acc) in enumerate(results.items()):
            plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
        
        # Save the comparison plot
        comparison_path = os.path.join(VISUALIZATION_DIR, f"{timestamp}_naive_bayes_comparison.png")
        plt.savefig(comparison_path)
        print(f"Saved comparison plot to {comparison_path}")
        plt.show()

if __name__ == "__main__":
    main()