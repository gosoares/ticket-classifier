"""Evaluation metrics for the ticket classifier."""

import numpy as np
import plotly.figure_factory as ff
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate(
    y_true: list[str],
    y_pred: list[str],
    classes: list[str],
) -> dict:
    """
    Calculate evaluation metrics for classification results.

    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        classes: List of valid class names (for ordering)

    Returns:
        Dict with accuracy, f1_macro, confusion_matrix, and report
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes),
        "report": classification_report(y_true, y_pred, labels=classes, zero_division=0),
    }


def print_report(metrics: dict, classes: list[str]) -> None:
    """
    Print formatted evaluation report.

    Args:
        metrics: Dict returned by evaluate()
        classes: List of class names for display
    """
    print("=" * 60)
    print("RELATÓRIO DE AVALIAÇÃO")
    print("=" * 60)

    print(f"\nAccuracy:    {metrics['accuracy']:.4f}")
    print(f"F1 Macro:    {metrics['f1_macro']:.4f}")

    print("\n" + "-" * 60)
    print("Classification Report:")
    print("-" * 60)
    print(metrics["report"])


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot confusion matrix using Plotly.

    Args:
        cm: Confusion matrix array from sklearn
        classes: List of class names for axis labels
        title: Plot title
    """

    # Convert to list for plotly
    cm_list = cm.tolist()

    # Create annotated heatmap
    fig = ff.create_annotated_heatmap(
        z=cm_list,
        x=classes,
        y=classes,
        colorscale="Blues",
        showscale=True,
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="True",
        xaxis={"side": "bottom"},
    )

    # Reverse y-axis to match sklearn convention
    fig.update_yaxes(autorange="reversed")

    fig.show()
