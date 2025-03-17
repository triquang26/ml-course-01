import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import DataLoader
from medmnist import PneumoniaMNIST
from torchvision import transforms
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.data.preprocess.neural_network import *
from src.features.features_selection.neural_network import *
MODEL_DIR = "trained" 
def compute_mean_std(dataset):
    """Compute dataset mean and standard deviation"""
    loader = DataLoader(dataset, batch_size=500, shuffle=False, num_workers=2)
    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, -1)
        mean += images.mean(dim=1).sum().item()
        std += images.std(dim=1).sum().item()
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean, std

def train_epoch(model, optimizer, data_loader, loss_history):
    """Train the model for one epoch"""
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target.squeeze(1))
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())

def evaluate(model, data_loader, loss_history):
    """Evaluate the model performance"""
    model.eval()
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target.squeeze(1), reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target.squeeze(1)).sum()

    avg_loss = total_loss / total_samples
    accuracy = correct_samples.item() / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * accuracy) + '%)\n')
    
    return accuracy, correct_samples.item(), total_samples


# def plot_confusion_matrix(model, data_loader, save_path=None):
#     """Plot confusion matrix for the model predictions"""
#     model.eval()
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for data, target in data_loader:
#             output = model(data)
#             target = target.squeeze(1)
#             _, preds = torch.max(output, dim=1)
#             all_preds.extend(preds.numpy())
#             all_labels.extend(target.numpy())

#     cm = confusion_matrix(all_labels, all_preds)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=["Normal", "Pneumonia"], 
#                 yticklabels=["Normal", "Pneumonia"])
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title('Confusion Matrix')
    
#     if save_path:
#         plt.savefig(save_path)
#         print(f"Saved confusion matrix to {save_path}")
    
#     plt.close()
#     return all_preds, all_labels
def plot_confusion_matrix(all_preds,all_labels, save_path=None):
    """Plot confusion matrix for the model predictions"""
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Normal", "Pneumonia"], 
                yticklabels=["Normal", "Pneumonia"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    
    plt.close()
    return all_preds, all_labels
# def plot_f1_heatmap(model, data_loader, save_path=None):
#     """Plot F1 score heatmap for each class"""
#     model.eval()
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for data, target in data_loader:
#             output = model(data)
#             target = target.squeeze(1)
#             _, preds = torch.max(output, dim=1)
#             all_preds.extend(preds.numpy())
#             all_labels.extend(target.numpy())

#     f1_scores = f1_score(all_labels, all_preds, average=None)
#     class_labels = ["Normal (0)", "Pneumonia (1)"]
#     f1_matrix = np.expand_dims(f1_scores, axis=0)

#     plt.figure(figsize=(6, 2))
#     sns.heatmap(f1_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
#                 xticklabels=class_labels, yticklabels=['F1 Score'])
#     plt.xlabel('Class')
#     plt.title('F1-Score Heatmap (Normal vs Pneumonia)')
    
#     if save_path:
#         plt.savefig(save_path)
#         print(f"Saved F1 score heatmap to {save_path}")
    
#     plt.close()
def plot_f1_heatmap( all_preds,all_labels,save_path=None):
    """Plot F1 score heatmap for each class"""


    f1_scores = f1_score(all_labels, all_preds, average=None)
    class_labels = ["Normal (0)", "Pneumonia (1)"]
    f1_matrix = np.expand_dims(f1_scores, axis=0)

    plt.figure(figsize=(6, 2))
    sns.heatmap(f1_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                xticklabels=class_labels, yticklabels=['F1 Score'])
    plt.xlabel('Class')
    plt.title('F1-Score Heatmap (Normal vs Pneumonia)')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved F1 score heatmap to {save_path}")
    
    plt.close()

def run_vision_transformer(train_dataset, test_dataset, batch_size=100, test_batch_size=1000,
                          n_epochs=10, learning_rate=0.003):
    """Run Vision Transformer model training and evaluation"""
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    # Get input shape
    data_sample = next(iter(train_loader))[0]
    C, H, W = data_sample.shape[1], data_sample.shape[2], data_sample.shape[3]
    
    # ViT hyperparameters
    patch = 4  # Patch size (image dimensions must be divisible by this)
    F_out = len(np.unique(train_dataset.labels))  # Number of classes
    F_o = 64  # Output dimension in attention mechanism
    heads = 4  # Number of attention heads
    F_in = 64  # Input dimension to transformer
    mlp_dim = 128  # Hidden dimension in feed-forward network
    trans_depth = 6  # Number of transformer blocks
    
    print(f"Creating Vision Transformer with input shape: {C}x{H}x{W}, {patch}x{patch} patches")
    
    # Create model
    model = VIT(
        channel=C,
        image_h=H,
        image_w=W,
        patch=patch,
        F_in=F_in,
        F_o=F_o,
        F_out=F_out,
        heads=heads,
        trans_depth=trans_depth,
        mlp_dim=mlp_dim,
        mode="zero"  # Use CLS token for classification
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss_history, test_loss_history = [], []
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Training loop
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch: {epoch}/{n_epochs}")
        train_epoch(model, optimizer, train_loader, train_loss_history)
        accuracy, _, _ = evaluate(model, test_loader, test_loss_history)
    model.save_model(MODEL_DIR + "/ViT.pth")
    # Final evaluation
    print("\nFinal Evaluation:")
    accuracy, _, _ = evaluate(model, test_loader, test_loss_history)
    
    # Get predictions for visualization
    predictions = []
    actual_labels = []
    correct_samples = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            target = target.squeeze(1)
            _, preds = torch.max(output, dim=1)
            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(target.cpu().numpy())
            correct_samples += preds.eq(target).sum().item()
    plot_confusion_matrix(predictions, actual_labels) 
    return accuracy, predictions, actual_labels, model


def test_vision_transformer(test_dataset, model, test_batch_size=1000): 
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    total_samples = len(test_loader.dataset)
    predictions = []
    actual_labels = []
    correct_samples = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            target = target.squeeze(1)
            _, preds = torch.max(output, dim=1)
            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(target.cpu().numpy())
            correct_samples += preds.eq(target).sum().item()

    accuracy = correct_samples / total_samples
    return accuracy, predictions, actual_labels