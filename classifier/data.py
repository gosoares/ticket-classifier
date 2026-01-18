"""Data loading and sampling utilities for ticket classification."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = Path(__file__).parent.parent / "dataset.csv"


def load_dataset(path: Path = DATASET_PATH) -> tuple[pd.DataFrame, list[str]]:
    """Load the ticket dataset from CSV."""
    df = pd.read_csv(path)
    classes = sorted(df["Topic_group"].unique().tolist())
    return df, classes


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    random_state: int = 123,
) -> pd.DataFrame:
    """
    Create a stratified sample from the dataset.

    Args:
        df: DataFrame with 'Document' and 'Topic_group' columns
        n: Total number of samples to return
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with stratified sample
    """
    # Calculate fraction needed for desired sample size
    frac = n / len(df)

    # Use stratified split to get proportional representation
    _, sample = train_test_split(
        df,
        test_size=frac,
        stratify=df["Topic_group"],
        random_state=random_state,
    )

    return sample.reset_index(drop=True)


def train_test_split_stratified(
    df: pd.DataFrame,
    test_size: int = 200,
    random_state: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test sets with stratification.

    Args:
        df: DataFrame with 'Document' and 'Topic_group' columns
        test_size: Number of samples for test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)
    """
    test_frac = test_size / len(df)

    train_df, test_df = train_test_split(
        df,
        test_size=test_frac,
        stratify=df["Topic_group"],
        random_state=random_state,
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
