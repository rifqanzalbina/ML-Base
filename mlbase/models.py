import numpy as np
import pickle
from typing import Optional

class BaseModel:
    def save_model(self, filepath : str):
        """Save the model to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filepath : str) -> 'BaseModel':
        """Load the model from a file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

class LinearRegression(BaseModel):
    def __init__(self, learning_rate : float = 0.01, n_iterations : int = 1000):
        self.learning_rate = learning_rate
        self.n_integrations = n_iterations
        self.weights = None
        self.bias = None