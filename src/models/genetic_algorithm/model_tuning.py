from core import cnn_model, decision_tree_model, bayesian_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def train_decision_tree(x_train, y_train, max_depth=10):
    """Train a Decision Tree model"""
    print("\nTraining Decision Tree model...")
    model = decision_tree_model.get_decision_tree(random_state=42, max_depth=max_depth)
    model.fit(x_train, y_train)
    return model

def train_naive_bayes(x_train, y_train):
    """Train a Gaussian Naive Bayes model"""
    print("\nTraining Gaussian Naive Bayes model...")
    model = bayesian_model.get_naive_bayes()
    model.fit(x_train, y_train)
    return model

def train_cnn(x_train, y_train, n_epochs=7, batch_size=32, learning_rate=0.001):
    """Train a simple CNN model"""
    print("\nTraining CNN model...")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = cnn_model.SimpleCNN().to(device)
    
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