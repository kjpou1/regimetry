import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer

def inspect_transformed_output(preprocessor: ColumnTransformer, transformed_array, max_rows: int = 5):
    """
    Utility to inspect a transformed NumPy or sparse matrix produced by a ColumnTransformer.

    Args:
        preprocessor (ColumnTransformer): The fitted transformer object.
        transformed_array (np.ndarray or sparse matrix): Output from .transform() or .fit_transform().
        max_rows (int): How many rows to display (default = 5).

    Prints:
        - Array type (dense/sparse)
        - Shape (rows, columns)
        - Feature names
        - Sample rows
    """
    try:
        # Check type and shape
        is_sparse = issparse(transformed_array)
        print(f"🧠 Transformed type: {'sparse' if is_sparse else 'dense'}")
        print(f"📐 Shape: {transformed_array.shape}")

        # Convert to dense if needed
        array_dense = transformed_array.toarray() if is_sparse else transformed_array

        # Try to get feature names
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            print("⚠️ get_feature_names_out() not supported in your scikit-learn version.")
            feature_names = [f"col_{i}" for i in range(array_dense.shape[1])]

        # Create DataFrame and show preview
        df = pd.DataFrame(array_dense, columns=feature_names)
        print("\n🔍 Sample Transformed Output:")
        print(df.head(max_rows))

    except Exception as e:
        print(f"❌ Error during inspection: {e}")



import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer

def print_feature_block(
    preprocessor: ColumnTransformer,
    transformed_array,
    pattern: str = None,
    max_rows: int = 10,
    max_cols: int = 100,
):
    """
    Prints a filtered block of transformed features from a pipeline output.

    Args:
        preprocessor (ColumnTransformer): The fitted transformer object.
        transformed_array (np.ndarray or sparse matrix): Transformed data.
        pattern (str, optional): If specified, filters columns containing this substring.
        max_rows (int): Number of rows to print.
        max_cols (int): Max columns to display in console.
    """
    try:
        # Convert sparse to dense if needed
        dense_array = transformed_array.toarray() if issparse(transformed_array) else transformed_array

        # Get feature names
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            feature_names = [f"col_{i}" for i in range(dense_array.shape[1])]

        # Create DataFrame
        df = pd.DataFrame(dense_array, columns=feature_names)

        # Apply column filter if provided
        if pattern:
            df = df.loc[:, df.columns.str.contains(pattern)]
            print(f"🔎 Showing columns matching pattern: '{pattern}'")

        # Set display limits
        pd.set_option("display.max_columns", max_cols)
        pd.set_option("display.width", None)

        # Print sample
        print(df.head(max_rows))

    except Exception as e:
        print(f"❌ Error in print_feature_block: {e}")