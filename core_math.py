import numpy as np

def dot_product(x, y):
    """Computes the dot product of two vectors."""
    return np.dot(x, y)

def vector_addition(x, y):
    """Adds two vectors."""
    return np.add(x, y)

def scalar_multiplication(scalar, vector):
    """Multiplies a vector by a scalar."""
    return np.multiply(scalar, vector)

def euclidean_norm(vector):
    """Computes the Euclidean norm (length) of a vector."""
    return np.linalg.norm(vector)
