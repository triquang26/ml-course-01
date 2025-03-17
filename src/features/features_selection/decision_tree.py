import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from src.data.preprocess.decision_tree import load_data

def apply_pca(train_features, test_features, n_components=50):
    print(f"Applying PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    
    train_reduced = pca.fit_transform(train_features)
    test_reduced = pca.transform(test_features)
    
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    print(f"PCA explained variance: {explained_variance:.2f}%")
    
    return train_reduced, test_reduced

def calculate_feature_correlation(features):
    correlation_matrix = np.corrcoef(features.T)
    return correlation_matrix

def select_correlated_features(correlation_matrix, threshold=0.5):
    rows, cols = np.triu_indices_from(correlation_matrix, k=1)
    
    correlated_pairs = []
    for i, j in zip(rows, cols):
        if abs(correlation_matrix[i, j]) >= threshold:
            correlated_pairs.append((i, j, correlation_matrix[i, j]))
    
    print(f"Found {len(correlated_pairs)} highly correlated feature pairs (≥{threshold})")
    return correlated_pairs

def visualize_confusion_matrix(y_true, y_pred, save_path=None, title='Confusion Matrix'):
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
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    return acc

def train_decision_tree():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    n_components = 50  
    X_train_pca, X_test_pca = apply_pca(X_train, X_test, n_components=n_components)

    correlation_matrix = calculate_feature_correlation(X_train_pca)
    correlated_features = select_correlated_features(correlation_matrix, threshold=0.5)

    X_train_final = np.vstack((X_train_pca, X_val))
    y_train_final = np.hstack((y_train, y_val))

    clf = DecisionTreeClassifier(max_depth=10, random_state=42)
    clf.fit(X_train_final, y_train_final)

    y_pred = clf.predict(X_test_pca)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    visualize_confusion_matrix(y_test, y_pred, save_path="figures/decision_tree_confusion_matrix.png")

if __name__ == "__main__":
    train_decision_tree()
