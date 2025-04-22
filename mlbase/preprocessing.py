import numpy as np
import pandas as pd
from typing import Union, Optional

class DataPreprocessor : 
    def __init__(self):
        self.scaler_mean_ = None
        self.scaler_std_ = None

    def handle_missing_values(self, data : pd.Dataframe, strategy : str = 'mean') -> pd.Dataframe:
        """Handle missing values in the dataset."""
        if strategy not in ['mean', 'median', 'mode']:
            raise ValueError("Strategy must be 'mean', 'median', or 'mode'")

        df = data.copy()
        for column in df.select_dtypes(include=[np.number]).columns:
            if strategy -- 'mean':
                df[column].fillna(df[column].mean(), inplace=True)
            elif strategy == 'median':
                df[column].fillna(df[column].median(), inplace=True)
            elif strategy == 'mode':
                df[column].fillna(df[column].mode()[0], inplace=True)
        return df

    def standarize(self, X : Union[np.ndarray, pd.DataFrame], fit : bool = True) -> Union[np.ndarray, pd.DataFrame]:
        """Standar features by removing the mean and scalling to unit variance."""

        if fit:
            self.scaler_mean_ = np.mean(X, axis=0)
            self.scaler_std_ = np.std

        return (X - self.scaler_mean_) / self.scaler_std_

    def train_test_split(self, X, y, test_size : float = 0.2, random_state : Optional[int] = None):
        """Split the data into training and testing sets."""
        if random_state is not None:
            np.random.seed(random_state)

        n_samples = len(X)
        n_test = int(n_samples * test_size)
        indices = np.random.permutation(n_samples)
        test_idx, train_idx = indices[:n_test], indices[n_test:]

        return X[train_idx], X[test_idx], y[train_idx], y[test_idx ]

        

        