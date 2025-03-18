import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.decomposition import PCA
from hmmlearn import hmm

def run_hidden_markov_model(train_dataset, test_dataset, n_components=50, n_bins=10, n_hidden_states=8):
    """
    Run a Hidden Markov Model on the PneumoniaMNIST dataset.
    
    Args:
        train_dataset: Training dataset from MedMNIST
        test_dataset: Test dataset from MedMNIST
        n_components: Number of PCA components to use
        n_bins: Number of bins for discretization
        n_hidden_states: Number of hidden states in the HMM
        
    Returns:
        accuracy: Model accuracy on test set
        predictions: Model predictions on test set
        actual_labels: True labels for test set
        model_dict: Dictionary of trained models
        preprocessors: Dictionary of preprocessors used
    """
    # Data preparation
    print("Preparing data for Hidden Markov Model...")
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

    print("Discretized data created:")
    print(f"Training data shape: {x_train_discrete.shape}")
    print(f"Test data shape: {x_test_discrete.shape}")

    # Define Hidden Markov Model structure
    print("Defining Hidden Markov Model structure...")

    classes = np.unique(y_train_discrete)
    n_classes = len(classes)

    # Train separate HMM for each class
    models = {}
    start_time = time.time()
    print("\nTraining HMM models for each class (this may take a while)...")

    for c in classes:
        print(f"Training model for class {c}...")
        class_data = x_train_discrete[y_train_discrete == c]
        
        model = hmm.GaussianHMM(
            n_components=n_hidden_states,
            covariance_type="diag",
            n_iter=20,
            init_params="",
            params="mct",
            random_state=42
        )
        
        model.startprob_ = np.ones(n_hidden_states) / n_hidden_states
        model.transmat_ = np.ones((n_hidden_states, n_hidden_states)) / n_hidden_states
        
        try:
            lengths = [n_components] * class_data.shape[0]
            flat_data = class_data.reshape(-1, 1)
            
            model.fit(flat_data, lengths)
            models[c] = model
            print(f"Model for class {c} trained successfully")
        except Exception as e:
            print(f"Error training model for class {c}: {e}")
            
            try:
                print("Trying simpler approach...")
                means = np.mean(class_data, axis=0)
                covs = np.cov(class_data, rowvar=False)
                
                model = hmm.GaussianHMM(
                    n_components=1,
                    covariance_type="full",
                    n_iter=5
                )
                model.startprob_ = np.array([1.0])
                model.transmat_ = np.array([[1.0]])
                model.means_ = means.reshape(1, -1)
                
                model.covars_ = np.diag(np.diag(covs)).reshape(1, n_components, n_components)
                
                models[c] = model
                print(f"Simple model for class {c} created")
            except Exception as e2:
                print(f"Simple approach failed: {e2}")

    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")

    # Make predictions
    start_time = time.time()
    print("\nMaking predictions on test data...")
    predictions = []

    for i in range(len(x_test_discrete)):
        sample = x_test_discrete[i]
        
        log_probs = {}
        for c, model in models.items():
            try:
                sample_reshaped = sample.reshape(-1, 1)
                log_probs[c] = model.score(sample_reshaped, [len(sample)])
            except Exception as e:
                try:
                    mean_vec = model.means_[0]
                    log_probs[c] = -np.sum((sample - mean_vec) ** 2)
                except:
                    print(f"Error in prediction for sample {i}, class {c}: {e}")
                    log_probs[c] = -np.inf
        
        if log_probs:
            predictions.append(max(log_probs, key=log_probs.get))
        else:
            predictions.append(np.random.choice(classes))

    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.2f} seconds for {len(x_test_discrete)} samples")

    # Evaluate results
    actual_labels = y_test_discrete
    accuracy = accuracy_score(actual_labels, predictions)
    print(f"\nAccuracy on test data: {accuracy:.4f}")
    
    # Package all relevant data for persistence
    preprocessors = {
        'pca': pca,
        'discretizer': discretizer,
        'params': {
            'n_components': n_components,
            'n_bins': n_bins,
            'n_hidden_states': n_hidden_states
        }
    }
    
    # Instead of returning a single model, return the dictionary of models
    # This will help with proper serialization and loading
    return accuracy, predictions, actual_labels, models, preprocessors

def visualize_results(predictions, actual_labels, save_path=None):
    """Visualize the performance of the Hidden Markov Model with confusion matrix and classification report
    
    Args:
        predictions: Model predictions on test set
        actual_labels: True labels for test set
        save_path: Path to save the confusion matrix visualization (optional)
    """
    cm = confusion_matrix(actual_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Normal', 'Pneumonia'], 
               yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Hidden Markov Model')
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()

    print("\nClassification Report:")
    print(classification_report(actual_labels, predictions, target_names=['Normal', 'Pneumonia']))

    print("\nHidden Markov Model analysis completed!")