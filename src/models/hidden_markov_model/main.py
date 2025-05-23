import numpy as np
from medmnist import PneumoniaMNIST
from medmnist import INFO
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from hmmlearn.hmm import GaussianHMM
import seaborn as sns
import matplotlib.pyplot as plt
import wandb

wandb.init(
    project="pneumonia-hmm",
    config={
        "dataset": "PneumoniaMNIST",
        "model": "GaussianHMM",
        "n_components": 4,
        "max_iter": 100,
        "random_state": 42,
        "val_size": 0.1,
    }
)
cfg = wandb.config
train_ds = PneumoniaMNIST(split="train", download=True)
val_ds   = PneumoniaMNIST(split="val",   download=True)
test_ds  = PneumoniaMNIST(split="test",  download=True)
print (train_ds)
def ds_to_arrays(ds):
    imgs = np.stack([np.array(img) for img, _ in ds], axis=0)         # (N,28,28)
    labels = np.array([lbl for _, lbl in ds]).squeeze()              # (N,)
    # normalize & flatten
    imgs = imgs.astype(np.float32) / 255.0
    imgs = imgs.reshape(len(imgs), -1)                               # (N,784)
    return imgs, labels

X_train, y_train = ds_to_arrays(train_ds)
X_val,   y_val   = ds_to_arrays(val_ds)
X_test,  y_test  = ds_to_arrays(test_ds)

print(f"Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
print(f"Labels: train={np.bincount(y_train)}, val={np.bincount(y_val)}, test={np.bincount(y_test)}")

models = {}
for cls in np.unique(y_train):
    model = GaussianHMM(
        n_components=cfg.n_components,
        covariance_type="diag",
        n_iter=cfg.max_iter,
        random_state=cfg.random_state
    )
    X_cls = X_train[y_train == cls]
    lengths = [X_cls.shape[1]] * len(X_cls)
    model.fit(X_cls.reshape(-1,1), lengths)
    models[cls] = model
    wandb.log({
        f"transmat_class_{cls}": wandb.Histogram(model.transmat_),
        f"startprob_class_{cls}": wandb.Histogram(model.startprob_)
    })

    y_pred = []
for x in X_test:
    seq = x.reshape(-1,1)
    # score each class-model
    scores = {cls: m.score(seq) for cls,m in models.items()}
    y_pred.append(max(scores, key=scores.get))
y_pred = np.array(y_pred)

# 4) Metrics & plots
report = classification_report(y_test, y_pred, target_names=["normal","pneumonia"])
print(report)
wandb.log({"classification_report": report})

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix

def plot_hmm_f1_heatmap(models, X, y, class_labels=None):
    """
    Compute per‐class F1 for an HMM classifier and plot as a heatmap.

    models       : dict[label -> GaussianHMM]
    X            : np.ndarray of shape (N, D)  # D=784
    y            : np.ndarray of shape (N,)
    class_labels : list[str] of length n_classes
    """
    # 1) Predict
    preds = []
    for x in X:
        seq = x.reshape(-1, 1)
        # score each class-model, pick the best
        scores = {cls: m.score(seq) for cls, m in models.items()}
        preds.append(max(scores, key=scores.get))
    preds = np.array(preds)

    # 2) Compute F1 per class
    f1s = f1_score(y, preds, average=None)

    # 3) Plot heatmap
    n_classes = len(f1s)
    if class_labels is None:
        class_labels = [str(c) for c in sorted(models.keys())]

    f1_matrix = f1s.reshape(1, n_classes)
    plt.figure(figsize=(6, 2))
    sns.heatmap(
        f1_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=class_labels,
        yticklabels=["F1 Score"]
    )
    plt.xlabel("Class")
    plt.title("F1-Score Heatmap")
    plt.show()


def plot_hmm_confusion_matrix(models, X, y, class_labels=None):
    """
    Compute and plot confusion matrix for an HMM classifier.

    models       : dict[label -> GaussianHMM]
    X            : np.ndarray of shape (N, D)
    y            : np.ndarray of shape (N,)
    class_labels : list[str] of length n_classes
    """
    # 1) Predict
    preds = []
    for x in X:
        seq = x.reshape(-1, 1)
        scores = {cls: m.score(seq) for cls, m in models.items()}
        preds.append(max(scores, key=scores.get))
    preds = np.array(preds)

    # 2) Confusion matrix
    cm = confusion_matrix(y, preds, labels=sorted(models.keys()))
    if class_labels is None:
        class_labels = [str(c) for c in sorted(models.keys())]

    # 3) Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

plot_hmm_f1_heatmap(models, X_test, y_test, class_labels=["Normal (0)", "Pneumonia (1)"])
plot_hmm_confusion_matrix(models, X_test, y_test, class_labels=["Normal (0)", "Pneumonia (1)"])