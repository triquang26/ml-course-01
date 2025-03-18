import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.decomposition import PCA

def run_bayesian_network(train_dataset, test_dataset, n_components=50, n_bins=10):
    """
    Run a Bayesian Network model on the PneumoniaMNIST dataset.
    
    Args:
        train_dataset: Training dataset from MedMNIST
        test_dataset: Test dataset from MedMNIST
        n_components: Number of PCA components to use
        n_bins: Number of bins for discretization
        
    Returns:
        accuracy: Model accuracy on test set
        predictions: Model predictions on test set
        actual_labels: True labels for test set
        model: Trained Bayesian Network model
        preprocessors: Dictionary containing preprocessors used
    """
    # Data preparation
    print("Preparing data for Bayesian Network model...")
    x_train = train_dataset.imgs.astype('float32') / 255.0
    y_train = train_dataset.labels.flatten()

    x_test = test_dataset.imgs.astype('float32') / 255.0
    y_test = test_dataset.labels.flatten()

    # Flatten images
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    print(f"Training data shape: {x_train_flat.shape}")
    print(f"Test data shape: {x_test_flat.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")

    # Handle large datasets by subsampling if needed
    n_samples = 999999999999  # Effectively no limit
    if len(x_train_flat) > n_samples:
        idx = np.random.choice(len(x_train_flat), n_samples, replace=False)
        x_train_subset = x_train_flat[idx]
        y_train_subset = y_train[idx]
    else:
        x_train_subset = x_train_flat
        y_train_subset = y_train

    # Dimensionality reduction with PCA
    print(f"Reducing dimensionality with PCA to {n_components} components...")
    pca = PCA(n_components=n_components)
    x_train_reduced = pca.fit_transform(x_train_subset)
    x_test_reduced = pca.transform(x_test_flat)

    print(f"Reduced data shape: {x_train_reduced.shape}")
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")

    # Discretize features
    print("Discretizing features...")
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    x_train_discrete = discretizer.fit_transform(x_train_reduced)
    x_test_discrete = discretizer.transform(x_test_reduced)

    x_train_discrete = x_train_discrete.astype(int)
    x_test_discrete = x_test_discrete.astype(int)
    y_train_discrete = y_train_subset.astype(int)
    y_test_discrete = y_test.astype(int)

    # Create dataframes for pgmpy
    feature_names = [f'F{i}' for i in range(n_components)]
    column_names = feature_names + ['pneumonia']

    train_data = np.column_stack((x_train_discrete, y_train_discrete))
    train_df = pd.DataFrame(train_data, columns=column_names)

    test_data = np.column_stack((x_test_discrete, y_test_discrete))
    test_df = pd.DataFrame(test_data, columns=column_names)

    print("Dataframes created:")
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print("\nTraining data preview:")
    print(train_df.head())

    # Define Bayesian Network structure
    print("\nDefining Bayesian Network structure...")
    edges = []
    for feature in feature_names:
        edges.append(('pneumonia', feature))

    model = BayesianNetwork(edges)

    # Train the Bayesian Network
    start_time = time.time()
    print("\nEstimating model parameters (this may take a while)...")
    try:
        model.fit(train_df, estimator=BayesianEstimator, prior_type='BDeu')
        print("Model training completed using BayesianEstimator!")
    except Exception as e:
        print(f"Error with BayesianEstimator: {e}")
        print("Falling back to MaximumLikelihoodEstimator...")
        model.fit(train_df, estimator=MaximumLikelihoodEstimator)
        print("Model training completed using MaximumLikelihoodEstimator!")

    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")

    # Validate model
    try:
        model.check_model()
        print("Model is valid.")
    except Exception as e:
        print(f"Warning: Model check failed: {e}")

    # Display sample of learned probabilities
    print("\nSample of learned probabilities:")
    for cpd in model.get_cpds()[:2]:  
        print(f"CPD of {cpd.variable}:")
        print(cpd)

    # Inference
    inference = VariableElimination(model)

    # Make predictions
    start_time = time.time()
    print("\nMaking predictions on test data...")
    predictions = []

    max_test_samples = 99999999999  # Effectively no limit
    test_indices = range(min(len(test_df), max_test_samples))

    for i in test_indices:
        evidence = {f'F{j}': test_df.iloc[i][f'F{j}'] for j in range(n_components)}
        try:
            query_result = inference.map_query(variables=['pneumonia'], evidence=evidence)
            predictions.append(query_result['pneumonia'])
        except Exception as e:
            print(f"Error in prediction for sample {i}: {e}")
            predictions.append(np.random.choice([0, 1]))

    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.2f} seconds for {len(test_indices)} samples")

    # Evaluate results
    actual_labels = test_df.iloc[test_indices]['pneumonia'].values
    accuracy = accuracy_score(actual_labels, predictions)
    print(f"\nAccuracy on test data: {accuracy:.4f}")
    
    # Trả về thêm các preprocessors
    preprocessors = {
        'pca': pca,
        'discretizer': discretizer,
        'params': {'n_components': n_components, 'n_bins': n_bins}
    }
    
    return accuracy, predictions, actual_labels, model, preprocessors

def visualize_results(predictions, actual_labels, save_path=None, title="Bayesian Network"):
    """Visualize the performance of the model with confusion matrix and classification report
    
    Args:
        predictions: Model predictions on test set
        actual_labels: True labels for test set
        save_path: Path to save the confusion matrix visualization (optional)
        title: Title for the visualization
    """
    cm = confusion_matrix(actual_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {title}')
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()

    print("\nClassification Report:")
    print(classification_report(actual_labels, predictions, 
                              target_names=['Normal', 'Pneumonia']))

    print(f"\n{title} analysis completed!")