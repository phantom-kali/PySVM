import numpy as np
import pickle

def save_model(model, filepath):
    """Saves the SVM model to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    """Loads a previously saved SVM model."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def calculate_accuracy(y_true, y_pred):
    """Calculates the accuracy of the SVM on a given dataset."""
    accuracy = np.mean(y_true == y_pred)
    return accuracy

def log_metrics(metrics, filepath):
    """Logs training metrics to a file."""
    with open(filepath, 'a') as f:
        f.write(metrics + '\n')
