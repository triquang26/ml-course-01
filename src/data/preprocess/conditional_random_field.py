import numpy as np
from medmnist import PneumoniaMNIST, INFO

def load_data():
    train_dataset = PneumoniaMNIST(split='train', download=True)
    # Convert PIL Images to numpy arrays explicitly
    X_train = np.array([np.array(train_dataset[i][0]).reshape(28, 28) for i in range(len(train_dataset))])
    Y_train = np.array([train_dataset[i][1] for i in range(len(train_dataset))])

    test_dataset = PneumoniaMNIST(split='test', download=True)
    X_test = np.array([np.array(test_dataset[i][0]).reshape(28, 28) for i in range(len(test_dataset))])
    Y_test = np.array([test_dataset[i][1] for i in range(len(test_dataset))])

    # Flatten the images into feature vectors
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Convert flattened image to feature dictionary
    def image_to_features(image_flat):
        features = {}
        for i in range(784):
            features[f'pixel_{i}'] = float(image_flat[i])
        return features

    # Prepare data for sklearn-crfsuite
    X_train_crf = [[image_to_features(X_train_flat[i])] for i in range(X_train_flat.shape[0])]
    Y_train_crf = [[str(Y_train[i])] for i in range(Y_train.shape[0])]

    X_test_crf = [[image_to_features(X_test_flat[i])] for i in range(X_test_flat.shape[0])]
    Y_test_crf = [[str(Y_test[i])] for i in range(Y_test.shape[0])]

    return (X_train_crf, Y_train_crf, X_test_crf, Y_test_crf)