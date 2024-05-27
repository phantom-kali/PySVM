import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(filepath, delimiter=','):
    """Loads data from a CSV file."""
    return pd.read_csv(filepath, delimiter=delimiter)

def normalize_data(data):
    """Normalizes data to zero mean and unit variance."""
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def shuffle_data(data):
    """Shuffles data points."""
    return data.sample(frac=1).reset_index(drop=True)

def split_data(X, y, test_size=0.2):
    """Splits data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size)

def preprocess_text_data(data, text_column, label_column):
    """Preprocesses text data into numerical format using TF-IDF."""
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data[text_column])
    y = data[label_column].values
    return X.toarray(), y
