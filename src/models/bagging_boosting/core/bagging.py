import numpy as np
import pickle
from collections import Counter

class BaggingClassifier:
    """From-scratch Bagging classifier with debug prints."""
    def __init__(self, base_estimator, n_estimators=10, max_samples=1.0, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = np.random.RandomState(random_state)
        self.estimators_ = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        m = int(self.max_samples * n_samples) if self.max_samples <= 1 else int(self.max_samples)
        indices = self.random_state.choice(n_samples, size=m, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        """Train n_estimators on bootstrap samples, printing progress."""
        self.estimators_ = []
        n = X.shape[0]
        m = int(self.max_samples * n) if self.max_samples <= 1 else int(self.max_samples)
        print(f"[Bagging] Starting training: {self.n_estimators} estimators, each on {m} samples")
        for i in range(self.n_estimators):
            print(f"[Bagging] Training estimator {i+1}/{self.n_estimators}")
            X_samp, y_samp = self._bootstrap_sample(X, y)
            # deep copy of base_estimator
            estimator = pickle.loads(pickle.dumps(self.base_estimator))
            estimator.fit(X_samp, y_samp)
            self.estimators_.append(estimator)
        print("[Bagging] Training complete.\n")
        return self

    def predict(self, X):
        """Predict by majority vote"""
        all_preds = np.array([est.predict(X) for est in self.estimators_])  # (n_estimators, n_samples)
        maj_vote = []
        for preds in all_preds.T:
            maj_vote.append(Counter(preds).most_common(1)[0][0])
        return np.array(maj_vote)
