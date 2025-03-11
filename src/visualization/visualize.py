import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# #bayesian
# from src.models.bayesian.bernoulli_naive_bayes import visualize_results as bnb_visualize_original
# from src.models.bayesian.gaussian_naive_bayes import visualize_results as gn_visualize_original
# from src.models.bayesian.multinomial_naive_bayes import visualize_results as mnb_visualize_original
# #decision_tree
# from src.models.decision_tree.decision_tree import visualize_results as dt_visualize_original
# #genetic_algorithm
# #graphical_model
# from src.models.graphical_model.augmentated_naive_bayes_graphical import visualize_results as nbg_visualize_original
# from src.models.graphical_model.bayesian_network_graphical import visualize_results as bng_visualize_original
# from src.models.graphical_model.hidden_markov_graphical import visualize_results as hmg_visualize_original
# #neural_network
# from src.models.neural_network.neural_network import visualize_results as nn_visualize_original
VISUALIZATION_DIR = "figures"

# def bnb_visualize(*args, **kwargs):
#     return bnb_visualize_original(*args, **kwargs)

# def gn_visualize(*args, **kwargs):
#     return gn_visualize_original(*args, **kwargs)

# def mnb_visualize(*args, **kwargs):
#     return mnb_visualize_original(*args, **kwargs)

# def dt_visualize(*args, **kwargs):
#     return dt_visualize_original(*args, **kwargs)

# def nbg_visualize(*args, **kwargs):
#     return nbg_visualize_original(*args, **kwargs)

# def bng_visualize(*args, **kwargs):
#     return bng_visualize_original(*args, **kwargs)

# def hmg_visualize(*args, **kwargs):
#     return hmg_visualize_original(*args, **kwargs)

# def nn_visualize(*args, **kwargs):
#     return nn_visualize_original(*args, **kwargs)



def visualize_results(y_true, y_pred, model_name, save_path=None, show=True):
    """
    A common visualization function for all models that creates and displays:
    1. Confusion matrix
    2. Performance metrics (accuracy, precision, recall, F1-score)
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values
    y_pred : array-like
        Estimated targets as returned by a classifier
    model_name : str
        Name of the model for display purposes
    save_path : str, optional
        Path to save the visualization plot
    show : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    dict
        Dictionary containing performance metrics
    """
    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    # For binary classification
    if len(np.unique(y_true)) == 2:
        prec = precision_score(y_true, y_pred, average='binary')
        rec = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
    else:
        prec = precision_score(y_true, y_pred, average='weighted')
        rec = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title(f'{model_name} - Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Plot metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [acc, prec, rec, f1]
    
    ax2.bar(metrics, values, color='skyblue')
    ax2.set_title(f'{model_name} - Performance Metrics')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add values on top of bars
    for i, v in enumerate(values):
        ax2.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    # Return metrics dictionary
    metrics_dict = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }
    
    return metrics_dict

def compare_models(results, save_path=None, show=True):
    """
    Create a comparative visualization of multiple model results
    
    Parameters:
    -----------
    results : dict
        Dictionary with model names as keys and metrics dictionaries as values
    save_path : str, optional
        Path to save the visualization plot
    show : bool, default=True
        Whether to display the plot
    """
    model_names = list(results.keys())
    
    # Prepare metrics for comparison
    accuracies = [results[model]['accuracy'] for model in model_names]
    precisions = [results[model]['precision'] for model in model_names]
    recalls = [results[model]['recall'] for model in model_names]
    f1_scores = [results[model]['f1_score'] for model in model_names]
    
    # Set up bar positions
    x = np.arange(len(model_names))
    width = 0.2
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bars
    ax.bar(x - width*1.5, accuracies, width, label='Accuracy', color='skyblue')
    ax.bar(x - width/2, precisions, width, label='Precision', color='lightgreen')
    ax.bar(x + width/2, recalls, width, label='Recall', color='salmon')
    ax.bar(x + width*1.5, f1_scores, width, label='F1-Score', color='purple')
    
    # Add labels and legend
    ax.set_title('Model Performance Comparison')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add values on top of bars
    for i, model in enumerate(model_names):
        metrics = [accuracies[i], precisions[i], recalls[i], f1_scores[i]]
        positions = [i - width*1.5, i - width/2, i + width/2, i + width*1.5]
        
        for pos, val in zip(positions, metrics):
            ax.text(pos, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Comparison visualization saved to {save_path}")
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close()