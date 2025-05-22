import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.data.preprocess.bagging_boosting import load_data
from persistence import ModelPersistence
from predict import BaggingBoostingPredictor

class BagBoostTester:
    """Tester for one bagging/boosting model type."""
    def __init__(self, model_type):
        self.model_type = model_type
        # predictor will flatten & normalize internally
        self.predictor = BaggingBoostingPredictor(model_type=model_type)

    def evaluate(self, X_test, y_test):
        """Return (accuracy, predictions)."""
        # pass raw imgs—predictor does the rest
        preds, _ = self.predictor.predict(X_test)
        acc = accuracy_score(y_test, preds)
        return acc, preds


def visualize_confusion_matrix(y_true, y_pred, model_type, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_type} Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]),
                     ha='center', va='center',
                     color='white' if cm[i,j] > cm.max()/2 else 'black')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved confusion matrix to {save_path}")
    plt.close()


def test_models():
    _, test_dataset = load_data()
    X_test = test_dataset.imgs        # raw 0–255 images
    y_test = test_dataset.labels.flatten()

    out_dir = "reports/figures"
    os.makedirs(out_dir, exist_ok=True)

    model_types = [
        'base_tree',
        'bagging_dt',
        'adaboost_dt',
        'bagging_scratch',
        'adaboost_scratch',
        'sk_gb',         
        'gb_scratch',    
    ]

    results = {}
    for m in model_types:
        print(f"\nTesting {m} …")
        tester = BagBoostTester(m)
        try:
            acc, preds = tester.evaluate(X_test, y_test)
            print(f"Accuracy for {m}: {acc:.4f}")
            print(classification_report(y_test, preds, target_names=['Normal','Pneumonia']))
        except Exception as e:
            print(f"Error testing {m}: {e}")
            continue

        save_path = os.path.join(out_dir, f"test_{m}_cm.png")
        visualize_confusion_matrix(y_test, preds, m, save_path=save_path)
        results[m] = acc

    return results


if __name__ == "__main__":
    results = test_models()
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for m, acc in results.items():
        print(f"{m:20s} Accuracy = {acc:.4f}")
