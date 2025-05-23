import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from medmnist import PneumoniaMNIST

def load_data(n_components=50):
    """
    Purpose: Load PneumoniaMNIST dataset and preprocess with standardization and PCA.
    Input: n_components (int) - Number of PCA components to retain.
    Output: Tuple (X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test) - Preprocessed data.
    """
    train_dataset = PneumoniaMNIST(split='train', download=True)
    val_dataset = PneumoniaMNIST(split='val', download=True)
    test_dataset = PneumoniaMNIST(split='test', download=True)

    X_train = train_dataset.imgs
    y_train = train_dataset.labels.flatten()
    X_val = val_dataset.imgs
    y_val = val_dataset.labels.flatten()
    X_test = test_dataset.imgs
    y_test = test_dataset.labels.flatten()

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    print(f"Explained variance ratio by {n_components} components: {sum(pca.explained_variance_ratio_):.4f}")

    return X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test

def preprocess_data(X, scaler=None, pca=None, n_components=50):
    """
    Purpose: Preprocess input data with standardization and PCA for prediction.
    Input:
        - X: Input data (images).
        - scaler: StandardScaler object (optional, created if None).
        - pca: PCA object (optional, created if None).
        - n_components (int): Number of PCA components.
    Output: Tuple (X_processed, scaler, pca) - Preprocessed data, scaler, and PCA objects.
    """
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    if pca is None:
        pca = PCA(n_components=n_components)
        X_processed = pca.fit_transform(X)
    else:
        X_processed = pca.transform(X)
    return X_processed, scaler, pca

def train_svm_model(X_train, y_train, C=1.0, kernel='rbf'):
    """
    Purpose: Train an SVM model with specified parameters.
    Input:
        - X_train: Training features (PCA-transformed).
        - y_train: Training labels.
        - C (float): Regularization parameter.
        - kernel (str): Kernel type for SVM (e.g., 'rbf').
    Output: svm_model - Trained SVM model.
    """
    svm_model = SVC(C=C, kernel=kernel, random_state=42, probability=True)
    svm_model.fit(X_train, y_train)
    return svm_model

def predict_with_model(model, X, scaler, pca):
    """
    Purpose: Make predictions using a trained model on preprocessed data.
    Input:
        - model: Trained model.
        - X: Input data (images).
        - scaler: StandardScaler object.
        - pca: PCA object.
    Output: Tuple (predictions, probabilities) - Predicted labels and probabilities.
    """
    X_processed, _, _ = preprocess_data(X, scaler=scaler, pca=pca)
    predictions = model.predict(X_processed)
    try:
        probabilities = model.predict_proba(X_processed)
    except AttributeError:
        probabilities = None
    return predictions, probabilities

def predict_single_image(model, image, scaler, pca):
    """
    Purpose: Predict the class of a single image.
    Input:
        - model: Trained model.
        - image: Single image data.
        - scaler: StandardScaler object.
        - pca: PCA object.
    Output: Tuple (prediction, probability) - Predicted label and probability (if available).
    """
    if len(image.shape) == 2:
        image = image.reshape(1, -1)
    predictions, probabilities = predict_with_model(model, image, scaler, pca)
    return predictions[0], probabilities[0] if probabilities is not None else None