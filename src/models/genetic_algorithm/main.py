import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from medmnist import INFO, PneumoniaMNIST
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import genetic algorithm
from genetic_algorithm import GeneticAlgorithm

# Define visualization directory
VISUALIZATION_DIR = "reports/figures"

class SimpleCNN(nn.Module):
    """
    Simple CNN model for binary classification of pneumonia images.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input channels = 1, output channels = 32, kernel_size = 3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Reduces 28x28 -> 14x14
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def numpy_to_tensor(x):
    """Convert numpy array to PyTorch tensor with correct shape for CNN"""
    if x.ndim == 3:
        # x is (N, H, W) - add channel dimension to get (N, 1, H, W)
        x = np.expand_dims(x, axis=1)
    elif x.ndim == 4 and x.shape[-1] == 1:
        # x is (N, H, W, 1) - convert to (N, 1, H, W)
        x = x.transpose(0, 3, 1, 2)
    # Otherwise, assume it's already in the desired format
    return torch.tensor(x, dtype=torch.float32)

def load_data():
    """Load and preprocess the PneumoniaMNIST dataset"""
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
    
    # Preprocess images
    x_train = train_dataset.imgs.astype('float32') / 255.0
    y_train = train_dataset.labels.flatten()
    
    x_test = test_dataset.imgs.astype('float32') / 255.0
    y_test = test_dataset.labels.flatten()
    
    # Flatten images for traditional models
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    
    # Convert to tensor format for CNN
    x_train_tensor = numpy_to_tensor(x_train)
    x_test_tensor = numpy_to_tensor(x_test)
    
    # Split training data to create validation set for GA optimization
    print("Creating validation split...")
    (x_train_flat_model, x_val_flat, 
     x_train_tensor_model, x_val_tensor, 
     y_train_model, y_val) = train_test_split(
        x_train_flat, x_train_tensor, y_train, test_size=0.2, random_state=42)
    
    print(f"Model training samples: {len(y_train_model)}")
    print(f"Validation samples: {len(y_val)}")
    
    return (x_train_flat_model, x_val_flat, x_test_flat, 
            x_train_tensor_model, x_val_tensor, x_test_tensor,
            y_train_model, y_val, y_test)

def train_decision_tree(x_train, y_train, max_depth=10):
    """Train a Decision Tree model"""
    print("\nTraining Decision Tree model...")
    model = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    model.fit(x_train, y_train)
    return model

def train_naive_bayes(x_train, y_train):
    """Train a Gaussian Naive Bayes model"""
    print("\nTraining Gaussian Naive Bayes model...")
    model = GaussianNB()
    model.fit(x_train, y_train)
    return model

def train_cnn(x_train, y_train, n_epochs=7, batch_size=32, learning_rate=0.001):
    """Train a simple CNN model"""
    print("\nTraining CNN model...")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SimpleCNN().to(device)
    
    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create data loader
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    train_dataset = TensorDataset(x_train, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    return model, device

def evaluate_model(model, x_test, y_test, model_name):
    """Evaluate model performance"""
    if model_name == "CNN":
        # Handle CNN evaluation separately because it uses tensors
        device = x_test.device if hasattr(x_test, 'device') else torch.device('cpu')
        model.eval()
        with torch.no_grad():
            outputs = model(x_test).cpu().numpy().flatten()
        predictions = (outputs > 0.5).astype(int)
    else:
        # For scikit-learn models
        predictions = model.predict(x_test)
    
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    return accuracy, f1, cm, predictions

def get_probabilities(model, x_test, model_name, device=None):
    """Get probability predictions from a model"""
    if model_name == "CNN":
        # Handle CNN separately because it uses tensors
        model.eval()
        with torch.no_grad():
            x_test_device = x_test.to(device) if device else x_test
            prob = model(x_test_device).cpu().numpy().flatten()
    else:
        # For scikit-learn models
        prob = model.predict_proba(x_test)[:, 1]  # Get probability of positive class
    
    return prob

def visualize_results(model_names, accuracies, f1_scores, confusion_matrices, timestamp):
    """Visualize model comparison results"""
    # Create bar plot for accuracy and F1 scores
    plt.figure(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(model_names))
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy')
    plt.bar(x + width/2, f1_scores, width, label='F1 Score')
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Comparison - Accuracy and F1 Score')
    plt.xticks(x, model_names)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add text labels
    for i, acc in enumerate(accuracies):
        plt.text(i - width/2, acc + 0.01, f'{acc:.4f}', ha='center')
    for i, f1 in enumerate(f1_scores):
        plt.text(i + width/2, f1 + 0.01, f'{f1:.4f}', ha='center')
    
    # Save plot
    comparison_path = os.path.join(VISUALIZATION_DIR, f"{timestamp}_model_comparison.png")
    plt.savefig(comparison_path)
    print(f"Saved comparison plot to {comparison_path}")
    plt.close()
    
    # Create confusion matrix plots
    for i, (name, cm) in enumerate(zip(model_names, confusion_matrices)):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=["Normal", "Pneumonia"],
                   yticklabels=["Normal", "Pneumonia"])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{name} - Confusion Matrix')
        
        cm_path = os.path.join(VISUALIZATION_DIR, f"{timestamp}_{name.lower().replace(' ', '_')}_cm.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"Saved {name} confusion matrix to {cm_path}")

def main():
    """Main function to run all models and ensemble"""
    # Create visualization directory if it doesn't exist
    ensure_dir_exists(VISUALIZATION_DIR)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    print(f"{'='*80}\nENSEMBLE MODEL WITH GENETIC ALGORITHM OPTIMIZATION\n{'='*80}")
    
    # Load and preprocess data
    (x_train_flat, x_val_flat, x_test_flat,
     x_train_tensor, x_val_tensor, x_test_tensor,
     y_train, y_val, y_test) = load_data()
    
    # Train individual models
    dt_model = train_decision_tree(x_train_flat, y_train)
    bayes_model = train_naive_bayes(x_train_flat, y_train)
    cnn_model, device = train_cnn(x_train_tensor, y_train)
    
    print("\nEvaluating models on validation set...")
    
    # Get validation probabilities for GA optimization
    dt_val_probs = get_probabilities(dt_model, x_val_flat, "Decision Tree")
    bayes_val_probs = get_probabilities(bayes_model, x_val_flat, "Naive Bayes")
    cnn_val_probs = get_probabilities(cnn_model, x_val_tensor, "CNN", device)
    
    # Run genetic algorithm to find optimal ensemble weights
    print("\nOptimizing ensemble weights with genetic algorithm...")
    ga = GeneticAlgorithm(pop_size=20, generations=30)
    best_weights, best_fitness = ga.run_ga(
        [dt_val_probs, cnn_val_probs, bayes_val_probs],
        y_val
    )
    
    print(f"\nOptimized Ensemble Weights: {[round(w, 3) for w in best_weights]}")
    print(f"Validation Fitness: {best_fitness:.4f}")
    
    # Evaluate individual models on test set
    print("\nEvaluating models on test set...")
    
    dt_test_probs = get_probabilities(dt_model, x_test_flat, "Decision Tree")
    bayes_test_probs = get_probabilities(bayes_model, x_test_flat, "Naive Bayes")
    cnn_test_probs = get_probabilities(cnn_model, x_test_tensor, "CNN", device)
    
    # Make ensemble prediction with optimized weights
    ensemble_test_probs = ga.ensemble_probabilities(
        best_weights,
        [dt_test_probs, cnn_test_probs, bayes_test_probs]
    )
    ensemble_test_preds = (ensemble_test_probs > 0.5).astype(int)
    
    # Evaluate results
    dt_test_preds = (dt_test_probs > 0.5).astype(int)
    bayes_test_preds = (bayes_test_probs > 0.5).astype(int)
    cnn_test_preds = (cnn_test_probs > 0.5).astype(int)
    
    model_names = ["Decision Tree", "Naive Bayes", "CNN", "GA Ensemble"]
    test_preds = [dt_test_preds, bayes_test_preds, cnn_test_preds, ensemble_test_preds]
    
    accuracies = []
    f1_scores = []
    confusion_matrices = []
    
    for name, preds in zip(model_names, test_preds):
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        
        accuracies.append(acc)
        f1_scores.append(f1)
        confusion_matrices.append(cm)
        
        print(f"\n{name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(model_names, accuracies, f1_scores, confusion_matrices, timestamp)
    
    print("\nEnsemble weights:")
    print(f"  Decision Tree: {best_weights[0]:.3f}")
    print(f"  CNN: {best_weights[1]:.3f}")
    print(f"  Naive Bayes: {best_weights[2]:.3f}")
    
    print("\nComplete!")

if __name__ == "__main__":
    main()