import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

def apply_pca(train_features, test_features, n_components=50):
    """Apply PCA for dimensionality reduction"""
    print(f"Applying PCA to reduce dimensions to {n_components} components...")
    pca = PCA(n_components=n_components)
    
    # Fit on training data and transform both training and test data
    train_reduced = pca.fit_transform(train_features)
    test_reduced = pca.transform(test_features)
    
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    print(f"PCA explained variance: {explained_variance:.2f}%")
    print(f"Reduced data shape - Train: {train_reduced.shape}, Test: {test_reduced.shape}")
    
    return train_reduced, test_reduced

def calculate_feature_correlation(features):
    """Calculate correlation between features"""
    correlation_matrix = np.corrcoef(features.T)
    return correlation_matrix

def select_correlated_features(correlation_matrix, threshold=0.5):
    """Find pairs of features with correlation above threshold"""
    # Get upper triangular indices (excluding diagonal)
    rows, cols = np.triu_indices_from(correlation_matrix, k=1)
    
    # Find pairs where correlation exceeds threshold
    correlated_pairs = []
    for i, j in zip(rows, cols):
        if abs(correlation_matrix[i, j]) >= threshold:
            correlated_pairs.append((i, j, correlation_matrix[i, j]))
    
    print(f"Found {len(correlated_pairs)} feature pairs with correlation >= {threshold}")
    return correlated_pairs

def visualize_confusion_matrix(y_true, y_pred, save_path=None, title='Confusion Matrix'):
    """Visualize confusion matrix for model results"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()
    
    # Calculate and return accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    return acc