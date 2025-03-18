import os
import pickle
from medmnist import PneumoniaMNIST
from pgmpy.models import BayesianNetwork

# Đường dẫn chuẩn cho thư mục lưu
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'trained'))

def ensure_dir_exists(directory_path):
    """Ensure a directory exists, creating it if necessary"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

class ModelPersistence:
    def __init__(self):
        self.models_dir = MODEL_DIR
        ensure_dir_exists(self.models_dir)
    
    @staticmethod
    def load_data(split='train'):
        """Load dataset split (train/test)"""
        try:
            dataset = PneumoniaMNIST(split=split, download=True)
            print(f"Loaded {split} dataset: {len(dataset)} samples")
            return dataset
        except Exception as e:
            print(f"Error loading {split} data: {e}")
            return None
        
    def save_model(self, model, model_name, preprocessors=None):
        """
        Save a trained model and its preprocessors to disk
        
        Args:
            model: Trained model object to save
            model_name: Name to use for the saved model
            preprocessors: Dictionary of preprocessing components (optional)
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Clean model name for consistency
            model_name_cleaned = model_name.lower().replace(' ', '_')
            
            # Save the model
            model_path = os.path.join(self.models_dir, f"{model_name_cleaned}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
            # Save preprocessors if provided
            if preprocessors is not None:
                # Save each preprocessor
                if 'pca' in preprocessors and preprocessors['pca'] is not None:
                    pca_path = os.path.join(self.models_dir, f"{model_name_cleaned}_pca.pkl")
                    with open(pca_path, 'wb') as f:
                        pickle.dump(preprocessors['pca'], f)
                        
                if 'discretizer' in preprocessors and preprocessors['discretizer'] is not None:
                    discretizer_path = os.path.join(self.models_dir, f"{model_name_cleaned}_discretizer.pkl")
                    with open(discretizer_path, 'wb') as f:
                        pickle.dump(preprocessors['discretizer'], f)
                
                # Save parameters if available
                if 'params' in preprocessors and preprocessors['params'] is not None:
                    params_path = os.path.join(self.models_dir, f"{model_name_cleaned}_params.pkl")
                    with open(params_path, 'wb') as f:
                        pickle.dump(preprocessors['params'], f)
                        
            print(f"Model {model_name} saved successfully")
            return True
            
        except Exception as e:
            print(f"Error saving model {model_name}: {str(e)}")
            return False
            
    def load_model(self, model_name):
        """Load saved model with proper validation"""
        model_name_cleaned = model_name.lower().replace(' ', '_')
        model_path = os.path.join(MODEL_DIR, f"{model_name_cleaned}.pkl")
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
            # Kiểm tra model có hợp lệ không
            if isinstance(model, BayesianNetwork):
                # Kiểm tra CPDs trong Bayesian Network
                if not model.get_cpds():
                    print(f"Cảnh báo: Mô hình {model_name} không có bảng CPD (Conditional Probability Distributions)")
                    return None
                    
            print(f"Mô hình được tải từ: {model_path}")
            return model
        except FileNotFoundError:
            print(f"Không tìm thấy mô hình tại: {model_path}")
            return None
        except Exception as e:
            print(f"Lỗi khi tải mô hình {model_name}: {str(e)}")
            return None
            
    @staticmethod
    def load_preprocessors(model_name):
        """Load saved preprocessors for a model"""
        model_name_cleaned = model_name.lower().replace(' ', '_')
        pca_path = os.path.join(MODEL_DIR, f"{model_name_cleaned}_pca.pkl")
        disc_path = os.path.join(MODEL_DIR, f"{model_name_cleaned}_discretizer.pkl")
        params_path = os.path.join(MODEL_DIR, f"{model_name_cleaned}_params.pkl")
        
        results = {}
        
        # Tải PCA nếu có
        try:
            with open(pca_path, 'rb') as f:
                results['pca'] = pickle.load(f)
            print(f"PCA loaded from: {pca_path}")
        except FileNotFoundError:
            print(f"No saved PCA found at: {pca_path}")
            results['pca'] = None
            
        # Tải discretizer nếu có
        try:
            with open(disc_path, 'rb') as f:
                results['discretizer'] = pickle.load(f)
            print(f"Discretizer loaded from: {disc_path}")
        except FileNotFoundError:
            print(f"No saved discretizer found at: {disc_path}")
            results['discretizer'] = None
            
        # Tải thông số bổ sung nếu có
        try:
            with open(params_path, 'rb') as f:
                results['params'] = pickle.load(f)
            print(f"Additional parameters loaded from: {params_path}")
        except FileNotFoundError:
            print(f"No saved parameters found at: {params_path}")
            results['params'] = None
            
        return results