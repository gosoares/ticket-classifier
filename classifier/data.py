"""Data loading and sampling utilities for ticket classification."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from classifier.config import RANDOM_STATE, VALIDATION_SIZE
from classifier.logging_config import get_logger

DATASET_PATH = Path(__file__).parent.parent / "dataset.csv"

logger = get_logger("data")


def load_dataset(path: Path = DATASET_PATH) -> tuple[pd.DataFrame, list[str]]:
    """Load the ticket dataset from CSV."""
    logger.info(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    classes = sorted(df["Topic_group"].unique().tolist())
    logger.info(f"Loaded {len(df):,} tickets with {len(classes)} classes")
    logger.debug(f"Classes: {classes}")
    return df, classes


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    random_state: int = RANDOM_STATE,
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
    test_size: int = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
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


def train_test_split_balanced(
    df: pd.DataFrame,
    test_size: int = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test sets with balanced test set.

    The test set will have the same number of samples from each class.
    This is useful for evaluating performance equally across all classes.

    Args:
        df: DataFrame with 'Document' and 'Topic_group' columns
        test_size: Total number of samples for test set (divided equally among classes)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)
    """
    n_classes = df["Topic_group"].nunique()
    n_per_class = test_size // n_classes

    logger.info(
        f"Splitting dataset: {n_per_class} samples per class ({n_classes} classes)"
    )

    # Sample n_per_class from each class
    test_df = df.groupby("Topic_group", group_keys=False).sample(
        n=n_per_class, random_state=random_state
    )
    train_df = df.drop(test_df.index)

    logger.info(f"Train: {len(train_df):,} tickets, Test: {len(test_df)} tickets")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def train_test_validation_split(
    df: pd.DataFrame,
    validation_size: int = VALIDATION_SIZE,
    train_size: float = 0.8,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, test, and validation sets.

    The validation set is a balanced sample with the same number of samples
    from each class. The remaining data is split into train/test with
    stratification.

    Args:
        df: DataFrame with 'Document' and 'Topic_group' columns
        validation_size: Total number of samples for validation set
        train_size: Fraction of remaining data to allocate to train set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df, validation_df)
    """
    n_classes = df["Topic_group"].nunique()
    if validation_size % n_classes != 0:
        raise ValueError(
            "validation_size must be divisible by the number of classes "
            f"({n_classes})"
        )

    n_per_class = validation_size // n_classes
    logger.info(
        f"Selecting validation set: {n_per_class} samples per class "
        f"({n_classes} classes)"
    )

    validation_df = df.groupby("Topic_group", group_keys=False).sample(
        n=n_per_class,
        random_state=random_state,
    )
    remaining_df = df.drop(validation_df.index)

    test_size = 1 - train_size
    train_df, test_df = train_test_split(
        remaining_df,
        test_size=test_size,
        stratify=remaining_df["Topic_group"],
        random_state=random_state,
    )

    logger.info(
        "Train/Test split on remaining data: "
        f"Train {len(train_df):,}, Test {len(test_df):,}, "
        f"Validation {len(validation_df):,}"
    )

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
    )
