"""Evaluation metrics for the ticket classifier."""

import numpy as np
import plotly.figure_factory as ff
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

import plotly.graph_objects as go

from classifier.logging_config import get_logger

logger = get_logger("metrics")


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
    logger.info(f"Calculating metrics for {len(y_true)} predictions")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    per_class = {
        class_name: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx, class_name in enumerate(classes)
    }
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_normalized = cm.astype(float)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm_normalized,
        row_sums,
        out=np.zeros_like(cm_normalized),
        where=row_sums != 0,
    )
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_normalized,
        "per_class": per_class,
        "report": classification_report(
            y_true, y_pred, labels=classes, zero_division=0
        ),
    }
    logger.info(
        f"Accuracy: {metrics['accuracy']:.4f}, F1 Macro: {metrics['f1_macro']:.4f}, "
        f"F1 Weighted: {metrics['f1_weighted']:.4f}, Kappa: {metrics['cohen_kappa']:.4f}, "
        f"MCC: {metrics['mcc']:.4f}"
    )
    return metrics


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

    print(f"\nAccuracy:      {metrics['accuracy']:.4f}")
    print(f"F1 Macro:      {metrics['f1_macro']:.4f}")
    print(f"F1 Weighted:   {metrics['f1_weighted']:.4f}")
    print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    print(f"MCC:           {metrics['mcc']:.4f}")

    print("\n" + "-" * 60)
    print("Classification Report:")
    print("-" * 60)
    print(metrics["report"])


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> None:
    """
    Plot confusion matrix using Plotly.

    Args:
        cm: Confusion matrix array from sklearn
        classes: List of class names for axis labels
        title: Plot title
        normalize: If True, normalize by row (true label) to show percentages
    """
    # Normalize if requested
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Handle division by zero
        # Format as percentages for annotations
        annotations = [[f"{val:.0%}" for val in row] for row in cm]
    else:
        annotations = [[str(int(val)) for val in row] for row in cm]

    # Convert to list for plotly
    cm_list = cm.tolist()

    # Create annotated heatmap
    fig = ff.create_annotated_heatmap(
        z=cm_list,
        x=classes,
        y=classes,
        annotation_text=annotations,
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


def plot_per_class_metrics(
    y_true: list[str],
    y_pred: list[str],
    classes: list[str],
    title: str = "Per-Class Metrics",
) -> None:
    """
    Plot per-class precision, recall, and F1 score as grouped bar chart.

    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        classes: List of class names for ordering
        title: Plot title
    """

    # Calculate per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    # Create grouped bar chart
    fig = go.Figure()

    fig.add_trace(
        go.Bar(name="Precision", x=classes, y=precision, marker_color="#1f77b4")
    )

    fig.add_trace(go.Bar(name="Recall", x=classes, y=recall, marker_color="#ff7f0e"))

    fig.add_trace(go.Bar(name="F1-Score", x=classes, y=f1, marker_color="#2ca02c"))

    fig.update_layout(
        title=title,
        xaxis_title="Class",
        yaxis_title="Score",
        barmode="group",
        yaxis=dict(range=[0, 1.05]),
        height=400,
    )

    fig.show()
