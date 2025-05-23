import time

def main():
    """
    Purpose: Orchestrate the training, evaluation, and testing of the SVM model with PCA.
    Input: None
    Output: None
    """
    ensure_dir_exists(VISUALIZATION_DIR)
    ensure_dir_exists(MODEL_DIR)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    n_components = 50

    print("\n" + "="*80)
    print("Running SVM model with PCA...")
    print("="*80)

    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(n_components=n_components)
    
    # Train model
    svm_model = train_svm_model(X_train, y_train, C=1.0, kernel='rbf')
    
    # Evaluate on validation set
    val_accuracy, val_predictions = evaluate_model(svm_model, X_val, y_val, dataset_name="Validation")
    
    # Evaluate on test set
    test_accuracy, test_predictions = evaluate_model(svm_model, X_test, y_test, dataset_name="Test")
    
    # Save model
    model_path = save_model(svm_model, model_name=f"svm_pca_{timestamp}.joblib")
    
    # Verify loading
    loaded_model = load_model(model_path)
    print("Successfully loaded the model")
    
    # Visualize results
    visualize_confusion_matrix(
        y_test,
        test_predictions,
        model_name="SVM with PCA",
        save_path=os.path.join(VISUALIZATION_DIR, f"{timestamp}_svm_pca_cm.png")
    )
    
    # Test model
    print("\nTesting model with PneumoniaPredictor...")
    test_accuracy, test_predictions, test_labels = test_model(model_path, n_components=n_components)
    
    results = {'SVM with PCA': test_accuracy}
    print(f"\nFinal Results: {results}")