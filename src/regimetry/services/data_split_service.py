from typing import Tuple

import pandas as pd


class DataSplitService:
    """
    A service for splitting datasets into train, validation, and test sets.
    Supports both sequential splitting (for time series data) and random splitting.
    """

    @staticmethod
    def sequential_split(
        df: pd.DataFrame, test_size: float, val_size: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits a dataset sequentially into train, validation, and test sets.

        Args:
            df (pd.DataFrame): The input DataFrame, indexed by time (datetime).
            test_size (float): Proportion of the data to include in the test set.
            val_size (float): Proportion of the remaining train data to include in the validation set.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test DataFrames.
        """
        # Calculate sizes
        n_test = int(len(df) * test_size)
        n_val = int((len(df) - n_test) * val_size)

        # Sequential splits
        train = df.iloc[: -n_test - n_val]
        val = df.iloc[-n_test - n_val : -n_test]
        test = df.iloc[-n_test:]

        return train, val, test

    @staticmethod
    def random_split(
        df: pd.DataFrame, test_size: float, val_size: float, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits a dataset randomly into train, validation, and test sets.

        Args:
            df (pd.DataFrame): The input DataFrame.
            test_size (float): Proportion of the data to include in the test set.
            val_size (float): Proportion of the remaining train data to include in the validation set.
            random_state (int): Random seed for reproducibility.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test DataFrames.
        """
        from sklearn.model_selection import train_test_split

        # Split into train+val and test
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=random_state
        )

        # Split train+val into train and validation
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val size proportion
        train, val = train_test_split(
            train_val, test_size=val_size_adjusted, random_state=random_state
        )

        return train, val, test

    @staticmethod
    def split_data(
        df: pd.DataFrame,
        method: str = "sequential",
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits a dataset into train, validation, and test sets using the specified method.

        Args:
            df (pd.DataFrame): The input DataFrame.
            method (str): The splitting method ("sequential" or "random").
            test_size (float): Proportion of the data to include in the test set.
            val_size (float): Proportion of the remaining train data to include in the validation set.
            random_state (int): Random seed for reproducibility (only used for random split).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test DataFrames.

        Raises:
            ValueError: If an unsupported splitting method is provided.
        """
        if method == "sequential":
            return DataSplitService.sequential_split(df, test_size, val_size)
        elif method == "random":
            return DataSplitService.random_split(df, test_size, val_size, random_state)
        else:
            raise ValueError(f"Unsupported splitting method: {method}")
