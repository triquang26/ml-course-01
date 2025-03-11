import numpy as np
from pgmpy.inference import VariableElimination
from .persistence import ModelPersistence
from sklearn.decomposition import PCA  # Add this import
from sklearn.preprocessing import KBinsDiscretizer
from hmmlearn import hmm
pca = PCA(n_components=50)
class ModelPredictor:
    def __init__(self):
        self.models = {}
        
    def load_model(self, model_name: str):
        if model_name == 'hidden_markov_model':
            try:
                # Load training data for fitting PCA
                train_data = ModelPersistence.load_data('train')
                X_train = train_data.imgs.reshape(train_data.imgs.shape[0], -1)
                X_train = X_train.astype('float32') / 255.0

                # Initialize and fit PCA
                self.pca = PCA(n_components=50)
                self.pca.fit(X_train)
                
                # Initialize other components
                self.discretizer = KBinsDiscretizer(
                    n_bins=10,
                    encode='ordinal',
                    strategy='quantile'
                )
                self.discretizer.fit(self.pca.transform(X_train))
                
                # Initialize HMM models
                self.models = {}
                for c in [0, 1]:
                    self.models[c] = hmm.GaussianHMM(
                        n_components=8,
                        covariance_type="diag",
                        n_iter=20,
                        random_state=42
                    )
                return True
                
            except Exception as e:
                print(f"Error loading HMM model: {e}")
                return False
        else:
            """Load a specific model"""
            model = ModelPersistence.load_model(model_name)
            if model is not None:
                self.models[model_name] = model
                return True
            return False
            
    def predict(self, X: np.ndarray, model_name: str) -> np.ndarray:
        if model_name == 'hidden_markov_model':
            # Apply preprocessing
            X_reduced = self.pca.transform(X)
            X_discrete = self.discretizer.transform(X_reduced)
            
            # Make predictions
            predictions = []
            for i in range(len(X_discrete)):
                sample = X_discrete[i].reshape(-1, 1)
                
                # Get log probability for each class
                log_probs = {}
                for c, model in self.models.items():
                    try:
                        log_probs[c] = model.score(sample, [len(sample)])
                    except:
                        log_probs[c] = float('-inf')
                
                # Predict class with highest probability
                if log_probs:
                    pred = max(log_probs, key=log_probs.get)
                else:
                    pred = 0  # Default prediction
                predictions.append(pred)
                
            return np.array(predictions)
        else:
            """Make predictions using the specified model"""
            # Basic preprocessing
            X = X.reshape(X.shape[0], -1)  # Flatten images
            X = X.astype('float32') / 255.0  # Normalize
            
            # Get model
            model = self.models.get(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not found")

            # Make predictions
            if model_name in ['bayesian_network', 'augmented_naive_bayes']:
                predictions = self._predict_bayesian(X, model)
            elif model_name == 'hidden_markov':
                predictions = self._predict_hmm(X, model)
            else:
                raise ValueError(f"Unknown model type: {model_name}")

            return predictions

    def _predict_bayesian(self, X: np.ndarray, model) -> np.ndarray:
        """Predict using Bayesian Network models"""
        predictions = []
        inference = VariableElimination(model)
        
        for i in range(len(X)):
            # Convert features to discrete values (0-9 range)
            discrete_features = np.digitize(X[i], bins=np.linspace(0, 1, 10)) - 1
            evidence = {f'F{j}': int(val) for j, val in enumerate(discrete_features)}
            
            try:
                query = inference.query(variables=['pneumonia'], evidence=evidence)
                pred = int(query.values.argmax())
                predictions.append(pred)
            except Exception as e:
                print(f"Error in prediction {i}: {e}")
                predictions.append(0)
                
        return np.array(predictions)
        
    def _predict_hmm(self, X: np.ndarray, model) -> np.ndarray:
        """Predict using HMM"""
        predictions = []
        
        for i in range(len(X)):
            # Convert features to discrete values
            discrete_features = np.digitize(X[i], bins=np.linspace(0, 1, 10)) - 1
            sample = discrete_features.reshape(-1, 1)
            
            log_probs = {}
            for c, m in model.items():
                try:
                    log_probs[c] = m.score(sample)
                except:
                    log_probs[c] = -np.inf
                    
            pred = max(log_probs, key=log_probs.get)
            predictions.append(pred)
            
        return np.array(predictions)