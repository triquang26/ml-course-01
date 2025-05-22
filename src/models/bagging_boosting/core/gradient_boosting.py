import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingClassifier:
    """
    From‐scratch gradient boosting for binary classification with
    deviance loss (logistic regression) + simple subsampling.
    """
    def __init__(
        self,
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        random_state=None,
        min_samples_leaf=1
    ):
        self.n_estimators   = n_estimators
        self.learning_rate  = learning_rate
        self.max_depth      = max_depth
        self.subsample      = subsample
        self.random_state   = random_state
        self.min_samples_leaf = min_samples_leaf

        # will hold our sequence of fitted trees and their weights
        self.estimators_    = []
        self.learning_rates_= []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        N, _ = X.shape

        # initial raw prediction F0 = 0
        F = np.zeros(N, dtype=float)

        for m in range(self.n_estimators):
            # probability estimates via logistic link
            prob = 1 / (1 + np.exp(-2 * F))
            # residuals = gradient of deviance loss
            residuals = y - prob

            # subsample indices (without replacement)
            if self.subsample < 1.0:
                idx = rng.choice(N, size=int(self.subsample * N), replace=False)
                X_train, r_train = X[idx], residuals[idx]
            else:
                idx = np.arange(N)
                X_train, r_train = X, residuals

            # build a new regression tree on the residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            tree.fit(X_train, r_train)

            gamma = 1.0

            # update raw predictions F(x) ← F(x) + ν * γ * fm(x)
            F += self.learning_rate * gamma * tree.predict(X)

            self.estimators_.append(tree)
            self.learning_rates_.append(self.learning_rate * gamma)

            resid_std = residuals.std()
            print(f"[GB] Stage {m+1}/{self.n_estimators}, resid_std={resid_std:.4f}")

    def predict_proba(self, X):
        # aggregate the sequence of trees
        F = np.zeros(X.shape[0], dtype=float)
        for tree, lr in zip(self.estimators_, self.learning_rates_):
            F += lr * tree.predict(X)

        # logistic link back to probability
        p = 1 / (1 + np.exp(-2 * F))
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)
