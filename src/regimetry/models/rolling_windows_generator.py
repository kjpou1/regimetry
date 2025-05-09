import numpy as np

class RollingWindowGenerator:
    """
    Generates overlapping rolling windows from a 2D NumPy array (post-transformation).
    
    Example:
        Input shape: (num_samples, num_features)
        Output shape: (num_windows, window_size, num_features)
    """

    def __init__(self, data: np.ndarray, window_size: int = 30, stride: int = 1):
        """
        Args:
            data (np.ndarray): The transformed dataset (2D array).
            window_size (int): Number of rows per rolling window.
            stride (int): Step size between windows.
        """
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.windows = None

    def generate(self) -> np.ndarray:
        """
        Generates rolling windows from the input data.

        Returns:
            np.ndarray: 3D array of shape (num_windows, window_size, num_features)
        """
        num_windows = (len(self.data) - self.window_size) // self.stride + 1
        self.windows = np.array([
            self.data[i:i + self.window_size]
            for i in range(0, num_windows * self.stride, self.stride)
        ])
        return self.windows

    def save(self, path: str):
        """
        Saves the generated windows to disk.

        Args:
            path (str): Output file path (e.g., 'EUR_USD_windows.npy')
        """
        if self.windows is None:
            raise ValueError("No windows to save. Run .generate() first.")
        np.save(path, self.windows)

    def load(self, path: str) -> np.ndarray:
        """
        Loads previously saved windows from disk.

        Args:
            path (str): Path to .npy file.

        Returns:
            np.ndarray: Loaded window array.
        """
        self.windows = np.load(path)
        return self.windows
