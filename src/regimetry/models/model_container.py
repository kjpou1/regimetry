import pickle

from sklearn.pipeline import Pipeline


class ModelContainer:
    def __init__(self, model, transformer):
        self._model = model
        self._transformer = transformer

    @property
    def model(self):
        """Get the trained model."""
        return self._model

    @property
    def transformer(self):
        """Get the ColumnTransformer."""
        return self._transformer

    def save(self, filepath):
        """Save both model and transformer together."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """Load the saved ModelContainer object."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
