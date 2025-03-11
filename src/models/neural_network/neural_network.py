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

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, V, Q, K, Fk, mask=None):
        C = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Fk, dtype=torch.float32))
        if mask is not None:
            C = C.masked_fill(mask == 0, float('-1e9'))
        C = nn.Softmax(dim=-1)(C)
        return torch.matmul(C, V)

class MultiHeadAttention(nn.Module):
    def __init__(self, Fv, Fk, Fv_o, Fk_o, heads, F_out, drop_out=0.1):
        super().__init__()
        self.Fv = Fv
        self.Fk = Fk
        self.Fv_o = Fv_o
        self.Fk_o = Fk_o
        self.heads = heads
        self.head_linearV = nn.Linear(self.Fv, self.Fv_o * self.heads)
        self.head_linearK = nn.Linear(self.Fk, self.Fk_o * self.heads)
        self.head_linearQ = nn.Linear(self.Fk, self.Fk_o * self.heads)
        self.final_linear = nn.Linear(heads * Fv_o, F_out)
        self.dropout = nn.Dropout(drop_out)
        self.attention = ScaledDotProductAttention()
        
    def forward(self, V, K, Q, mask=None):
        batch, seq_len, _ = V.shape
        v = self.head_linearV(V)
        k = self.head_linearK(K)
        q = self.head_linearQ(Q)

        v = v.reshape(batch, seq_len, self.heads, self.Fv_o).transpose(1, 2)
        k = k.reshape(batch, seq_len, self.heads, self.Fk_o).transpose(1, 2)
        q = q.reshape(batch, seq_len, self.heads, self.Fk_o).transpose(1, 2)

        attn = self.attention(v, q, k, self.Fk_o, mask)
        attn = attn.transpose(1, 2).reshape(batch, seq_len, self.heads * self.Fv_o)
        result = self.final_linear(attn)
        return result

class FeedForward(nn.Module):
    def __init__(self, F_in, F_o, dropout_rate=0.1):
        super().__init__()
        self.linear1 = nn.Linear(F_in, F_o)
        self.linear2 = nn.Linear(F_o, F_in)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, F_in, F_o, mlp_dim, heads, dropout_rate=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(F_in, eps=1e-6)
        self.norm = nn.LayerNorm(F_in, eps=1e-6)
        self.ff = FeedForward(F_in, mlp_dim, dropout_rate)
        self.attention = MultiHeadAttention(F_in, F_in, F_o, F_o, heads, F_in)
        
    def forward(self, x, mask=None):
        residue = x
        x = self.attn_norm(x)
        x = self.attention(x, x, x)
        x = x + residue
        residue = x
        x = self.norm(x)
        x = self.ff(x)
        x = x + residue
        return x

class VisionTransformer(nn.Module):
    def __init__(self, channel, image_h, image_w, patch, F_in, F_o, F_out, heads, trans_depth, mlp_dim, mode="average"):
        super().__init__()
        assert image_h % patch == 0, "Image height must be divisible by the patch size."
        assert image_w % patch == 0, "Image width must be divisible by the patch size."
        self.channel = channel
        self.image_h = image_h
        self.image_w = image_w
        self.patch = patch
        self.num_patch_h = image_h // patch
        self.num_patch_w = image_w // patch
        self.num_patch = self.num_patch_h * self.num_patch_w
        self.mode = mode

        patch_dim = channel * patch * patch
        self.patch_linear = nn.Linear(patch_dim, F_in)
        self.trans_layers = nn.ModuleList([TransformerBlock(F_in, F_o, mlp_dim, heads) for _ in range(trans_depth)])

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patch + 1, F_in))
        self.cls_token = nn.Parameter(torch.randn(1, 1, F_in))
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(F_in, eps=1e-6)
        self.final_linear = nn.Linear(F_in, F_out)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        x = x.reshape(b, c, self.num_patch_h, self.patch, self.num_patch_w, self.patch)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [b, num_patch_h, num_patch_w, c, patch, patch]
        x = x.reshape(b, self.num_patch, c * self.patch * self.patch)  # [b, num_patch, patch_dim]
        x = self.patch_linear(x)
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        for layer in self.trans_layers:
            x = layer(x)
        x = self.norm(x)
        x = self.final_linear(x)
        if self.mode == "average":
            return x.mean(dim=1)
        else:
            return x[:, 0, :]

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

def plot_confusion_matrix(model, data_loader, save_path=None):
    """Plot confusion matrix for the model predictions"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            target = target.squeeze(1)
            _, preds = torch.max(output, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(target.numpy())

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

def plot_f1_heatmap(model, data_loader, save_path=None):
    """Plot F1 score heatmap for each class"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            target = target.squeeze(1)
            _, preds = torch.max(output, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(target.numpy())

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
    model = VisionTransformer(
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
    
    # Final evaluation
    print("\nFinal Evaluation:")
    accuracy, _, _ = evaluate(model, test_loader, test_loss_history)
    
    # Get predictions for visualization
    predictions, actual_labels = plot_confusion_matrix(model, test_loader)  # Fixed: Added test_loader parameter
    
    return accuracy, predictions, actual_labels, model
def visualize_results(predictions, actual_labels, save_path=None):
    """Visualize the results using confusion matrix"""
    cm = confusion_matrix(actual_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Normal", "Pneumonia"],
                yticklabels=["Normal", "Pneumonia"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Vision Transformer - Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    
    plt.close()
    
    # Also calculate F1 scores
    f1_scores = f1_score(actual_labels, predictions, average=None)
    print(f"F1 Scores - Class 0 (Normal): {f1_scores[0]:.4f}, Class 1 (Pneumonia): {f1_scores[1]:.4f}")