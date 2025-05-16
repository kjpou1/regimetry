import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def attach_cluster_labels(df: pd.DataFrame, cluster_labels: np.ndarray, window_size: int) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Safely attach cluster labels to a DataFrame, aligned based on window size.
    Returns both the updated df and the actual assigned (trimmed) cluster_labels.

    Parameters:
    ----------
    df : pd.DataFrame
        The full regime input used to generate embeddings.
    cluster_labels : np.ndarray
        1D array of cluster assignments (raw, before trimming).
    window_size : int
        Rolling window size used to generate embeddings.

    Returns:
    -------
    tuple
        (df_with_cluster_column, cluster_labels_trimmed_aligned)
    """
    df = df.copy()
    offset = window_size - 1
    expected = len(df) - offset

    if len(cluster_labels) > expected:
        logger.warning(f"[attach_cluster_labels] ⚠️ Trimming cluster_labels from {len(cluster_labels)} to {expected}")
        cluster_labels = cluster_labels[-expected:]
    elif len(cluster_labels) < expected:
        raise ValueError(f"[attach_cluster_labels] ❌ Cluster label length mismatch: expected {expected}, got {len(cluster_labels)}")

    df["Cluster_ID"] = pd.NA
    df.loc[df.index[offset:offset + len(cluster_labels)], "Cluster_ID"] = pd.Series(cluster_labels, dtype="Int64").values

    logger.info(f"[attach_cluster_labels] ✅ Cluster labels aligned with offset={offset}, assigned={len(cluster_labels)}")

    return df




def verify_cluster_alignment(df: pd.DataFrame, window_size: int) -> None:
    """
    Verifies that cluster labels are correctly aligned to the DataFrame based on window size.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame expected to contain a 'Cluster_ID' column with assigned cluster labels.
    window_size : int
        The window size used to generate rolling embeddings.

    Raises:
    -------
    AssertionError or ValueError
        If alignment, length, or consistency checks fail.
    """
    offset = window_size - 1
    aligned_labels = df["Cluster_ID"].iloc[offset:]

    # Check presence and type
    assert "Cluster_ID" in df.columns, "[verify_cluster_alignment] ❌ 'Cluster_ID' column missing."

    # Check alignment length
    expected_length = len(df) - offset
    actual_length = aligned_labels.notna().sum()
    assert actual_length == expected_length, (
        f"[verify_cluster_alignment] ❌ Expected {expected_length} aligned cluster labels, got {actual_length}."
    )

    # Check for any NaNs
    if aligned_labels.isna().any():
        raise ValueError("[verify_cluster_alignment] ❌ Found NaNs in aligned Cluster_ID values.")

    logging.info("[verify_cluster_alignment] ✅ Cluster_ID alignment looks valid.")


def safe_numeric_column(df: pd.DataFrame, column: str, dtype=int) -> pd.Series:
    """
    Converts a nullable pandas column (e.g., Int64) into a NumPy-compatible type.
    Ensures compatibility with libraries like matplotlib and NumPy.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing the column.
    column : str
        Column name to convert.
    dtype : type
        Target Python-native numeric type, e.g., int or float.

    Returns:
    -------
    pd.Series
        Cleaned numeric column with pd.NA converted to np.nan or dropped.
    """
    series = df[column]
    if pd.api.types.is_integer_dtype(series):
        return series.astype(dtype, errors="ignore")
    return series

def infer_window_size(df_len: int, embeddings_len: int) -> int:
    return df_len - embeddings_len + 1
def normalize_cluster_id(cid):
    try:
        return str(int(float(cid)))
    except (ValueError, TypeError):
        return str(cid)  # fallback, won't crash
