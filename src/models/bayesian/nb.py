import numpy as np
from medmnist import INFO  # INFO for dataset metadata
import medmnist
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load the PneumoniaMNIST Dataset
# ----------------------------
dataset_info = INFO["pneumoniamnist"]
print(f"Dataset description: {dataset_info['description']}")
print(f"Number of classes: {len(dataset_info['label'])}, Labels: {dataset_info['label']}")

# Retrieve the dataset class using the info dictionary
DataClass = getattr(medmnist, dataset_info['python_class'])

# Load training and test sets
train_dataset = DataClass(split='train', download=True)
test_dataset  = DataClass(split='test', download=True)

# Extract images and labels from the datasets
X_train = train_dataset.imgs   # shape (4708, 28, 28)
y_train = train_dataset.labels # shape (4708, 1) or (4708, )
X_test  = test_dataset.imgs    # shape (624, 28, 28)
y_test  = test_dataset.labels  # shape (624, )

print("Training data shape:", X_train.shape, "Training labels shape:", y_train.shape)
print("Test data shape:", X_test.shape, "Test labels shape:", y_test.shape)

# ----------------------------
# 2. Preprocess Data for Each Naive Bayes Variant
# ----------------------------

# (A) Gaussian NB: Flatten and normalize pixel values to [0, 1]
X_train_gaussian = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
X_test_gaussian  = X_test.reshape(X_test.shape[0], -1).astype(np.float32)
X_train_gaussian /= 255.0
X_test_gaussian  /= 255.0

# (B) Bernoulli NB: Use binarized features from the normalized images.
# We'll threshold at 0.5.
X_train_bernoulli = (X_train_gaussian > 0.5).astype(np.int32)
X_test_bernoulli  = (X_test_gaussian > 0.5).astype(np.int32)

# (C) Multinomial NB: Use the original integer pixel values (0 to 255)
# We flatten the images and ensure they are integers.
X_train_multinomial = X_train.reshape(X_train.shape[0], -1).astype(np.int32)
X_test_multinomial  = X_test.reshape(X_test.shape[0], -1).astype(np.int32)

# Ensure labels are 1D arrays
y_train = y_train.reshape(-1)
y_test  = y_test.reshape(-1)

# ----------------------------
# 3. Train and Evaluate Each Model
# ----------------------------

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=['Normal', 'Pneumonia'])
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    return y_pred, acc, cm, cr

# --- Gaussian Naive Bayes ---
gnb = GaussianNB()
gnb.fit(X_train_gaussian, y_train)
evaluate_model(gnb, X_test_gaussian, y_test, "Gaussian Naive Bayes")

# --- Bernoulli Naive Bayes (Classic) ---
bnb = BernoulliNB()
bnb.fit(X_train_bernoulli, y_train)
evaluate_model(bnb, X_test_bernoulli, y_test, "Bernoulli Naive Bayes")

# --- Multinomial Naive Bayes ---
mnb = MultinomialNB()
mnb.fit(X_train_multinomial, y_train)
evaluate_model(mnb, X_test_multinomial, y_test, "Multinomial Naive Bayes")

# ----------------------------
# 4. (Optional) Visualize Confusion Matrices
# ----------------------------
models = ["Gaussian NB", "Bernoulli NB", "Multinomial NB"]
cms = [confusion_matrix(y_test, gnb.predict(X_test_gaussian)),
       confusion_matrix(y_test, bnb.predict(X_test_bernoulli)),
       confusion_matrix(y_test, mnb.predict(X_test_multinomial))]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, cm, title in zip(axes, cms, models):
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title)
    tick_marks = np.arange(len(dataset_info['label']))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(dataset_info['label'], rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(dataset_info['label'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center", color="red")
plt.tight_layout()
plt.show()
