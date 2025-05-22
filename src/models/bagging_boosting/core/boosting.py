import numpy as np
import pickle
import math

class AdaBoostClassifier:
    def __init__(self, base_estimator, n_estimators=10, learning_rate=1.0, random_state=None):
        self.base_estimator    = base_estimator
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.random_state      = np.random.RandomState(random_state)
        self.estimators_       = []
        self.estimator_weights_= []

    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples
        print(f"[AdaBoost] Starting training: {self.n_estimators} rounds")
        for m in range(self.n_estimators):
            print(f"[AdaBoost] Round {m+1}/{self.n_estimators}")
            # clone and train on weighted data, not by resampling
            estimator = pickle.loads(pickle.dumps(self.base_estimator))
            estimator.fit(X, y, sample_weight=w)

            # compute error on full set
            pred = estimator.predict(X)
            miss = (pred != y).astype(int)
            err  = np.dot(w, miss)

            if err <= 0 or err >= 0.5:
                print(f"[AdaBoost] stopping early at round {m+1}, error={err:.4f}")
                alpha = 1e9 if err == 0 else 0
                self.estimators_.append(estimator)
                self.estimator_weights_.append(alpha)
                break

            alpha = self.learning_rate * 0.5 * math.log((1 - err) / err)
            print(f"[AdaBoost] error={err:.4f} → alpha={alpha:.4f}")
            
            self.estimators_.append(estimator)
            self.estimator_weights_.append(alpha)
            
            # update and renormalize weights
            w *= np.exp(alpha * miss)
            w /= np.sum(w)
            print(f"[AdaBoost] Updated weights (first 10): {w[:10]}\n")

            self.estimators_.append(estimator)
            self.estimator_weights_.append(alpha)

        print("[AdaBoost] Training complete.\n")
        return self
    def predict(self, X):
        """
        Predict classes by weighted majority vote of the learners.
        """
        if not self.estimators_:
            raise ValueError("No estimators found. Did you call fit()?")

        # Collect predictions from each learner: shape (n_learners, n_samples)
        all_preds = np.array([est.predict(X) for est in self.estimators_])
        classes   = np.unique(all_preds)
        
        # Weighted vote accumulator: shape (n_classes, n_samples)
        agg = np.zeros((classes.shape[0], X.shape[0]))
        for alpha, preds in zip(self.estimator_weights_, all_preds):
            for idx, cls in enumerate(classes):
                agg[idx] += alpha * (preds == cls)

        # Choose class with highest aggregated weight
        final = classes[np.argmax(agg, axis=0)]
        return final