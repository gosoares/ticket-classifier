"""Runner module for executing the full evaluation pipeline."""

import json
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from classifier.config import K_SIMILAR, RANDOM_STATE, TEST_SIZE
from classifier.data import load_dataset, train_test_split_balanced
from classifier.graph import classify_ticket
from classifier.llm import ClassificationError, TicketClassifier, TokenUsage
from classifier.logging_config import get_logger, setup_logging
from classifier.metrics import evaluate
from classifier.rag import TicketRetriever

logger = get_logger("runner")


def run_evaluation(
    dataset_path: str | Path = "dataset.csv",
    output_dir: str | Path = "output",
    test_size: int = TEST_SIZE,
    k_similar: int = K_SIMILAR,
    random_state: int = RANDOM_STATE,
    model: str | None = None,
    use_references: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Execute the full evaluation pipeline.

    Steps:
    1. Load dataset and split into train/test
    2. Index training tickets for RAG retrieval
    3. Classify all test tickets
    4. Calculate evaluation metrics
    5. Save results to output files

    Args:
        dataset_path: Path to the CSV dataset file
        output_dir: Directory for output files
        test_size: Number of test samples (divided equally among classes)
        k_similar: Number of similar tickets to retrieve
        random_state: Random seed for reproducibility
        model: LLM model name (from LLM_MODEL env var if not specified)
        use_references: Whether to use reference tickets in prompts
        verbose: Enable verbose logging to terminal

    Returns:
        Dict with metrics and output file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(output_path, verbose=verbose)
    logger.info("=" * 60)
    logger.info("TICKET CLASSIFIER - EVALUATION PIPELINE")
    logger.info("=" * 60)

    # Log configuration
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Test size: {test_size}")
    logger.info(f"K similar: {k_similar}")
    logger.info(f"Model: {model}")
    logger.info(f"Use references: {use_references}")
    logger.info(f"Random state: {random_state}")

    # Step 1: Load and split data
    logger.info("-" * 60)
    logger.info("Step 1: Loading and splitting dataset")
    df, classes = load_dataset(Path(dataset_path))
    train_df, test_df = train_test_split_balanced(df, test_size, random_state)

    # Step 2: Index training data
    logger.info("-" * 60)
    logger.info("Step 2: Indexing training data for RAG")
    retriever = TicketRetriever()
    retriever.index(train_df)

    # Compute reference tickets if needed
    reference_tickets = None
    if use_references:
        reference_tickets = retriever.compute_representatives()

    # Step 3: Classify test tickets
    logger.info("-" * 60)
    logger.info(f"Step 3: Classifying {len(test_df)} test tickets")
    classifier = TicketClassifier(model=model)

    classifications = []
    errors = []
    y_true = []
    y_pred = []
    total_tokens = TokenUsage(0, 0, 0)

    pbar = tqdm(test_df.iterrows(), total=len(test_df), desc="Classifying")
    for _, row in pbar:
        ticket_text = row["Document"]
        true_label = row["Topic_group"]

        try:
            details = classify_ticket(
                ticket=ticket_text,
                retriever=retriever,
                classifier=classifier,
                classes=classes,
                reference_tickets=reference_tickets,
            )

            y_true.append(true_label)
            y_pred.append(details.result.classe)

            # Accumulate token usage
            total_tokens = total_tokens + details.token_usage

            # Update progress bar with last result
            is_correct = true_label == details.result.classe
            pbar.set_postfix(pred=details.result.classe, ok=is_correct)

            classifications.append(
                {
                    "ticket": ticket_text,
                    "true_class": true_label,
                    "predicted_class": details.result.classe,
                    "justification": details.result.justificativa,
                    "correct": is_correct,
                    "system_prompt": details.system_prompt,
                    "user_prompt": details.user_prompt,
                    "similar_tickets": details.similar_tickets,
                    "retries": details.retries,
                    "token_usage": {
                        "prompt_tokens": details.token_usage.prompt_tokens,
                        "completion_tokens": details.token_usage.completion_tokens,
                        "total_tokens": details.token_usage.total_tokens,
                    },
                }
            )

        except ClassificationError as e:
            pbar.set_postfix(error=e.reason[:20])
            errors.append(
                {
                    "ticket": ticket_text,
                    "true_class": true_label,
                    "reason": e.reason,
                    "raw_response": e.raw_response,
                }
            )

    logger.info(f"Classified: {len(classifications)}, Errors: {len(errors)}")
    logger.info(
        f"Total tokens: {total_tokens.total_tokens} "
        f"(prompt: {total_tokens.prompt_tokens}, completion: {total_tokens.completion_tokens})"
    )

    # Step 4: Calculate metrics
    logger.info("-" * 60)
    logger.info("Step 4: Calculating metrics")
    metrics = {}
    if y_true:
        metrics = evaluate(y_true, y_pred, classes)
    else:
        logger.warning("No successful classifications, skipping metrics")

    # Step 5: Save outputs
    logger.info("-" * 60)
    logger.info("Step 5: Saving results")

    timestamp = datetime.now().isoformat()

    # Save classifications report
    classifications_path = output_path / "classifications.json"
    classifications_report = {
        "metadata": {
            "timestamp": timestamp,
            "dataset": str(dataset_path),
            "test_size": test_size,
            "k_similar": k_similar,
            "model": model,
            "use_references": use_references,
            "random_state": random_state,
        },
        "summary": {
            "total": len(test_df),
            "classified": len(classifications),
            "errors": len(errors),
            "correct": sum(1 for c in classifications if c["correct"]),
            "token_usage": {
                "prompt_tokens": total_tokens.prompt_tokens,
                "completion_tokens": total_tokens.completion_tokens,
                "total_tokens": total_tokens.total_tokens,
            },
        },
        "classifications": classifications,
        "errors": errors,
    }
    with open(classifications_path, "w", encoding="utf-8") as f:
        json.dump(classifications_report, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved classifications to {classifications_path}")

    # Save metrics report
    metrics_path = output_path / "metrics.json"
    metrics_report = {
        "metadata": {
            "timestamp": timestamp,
            "dataset": str(dataset_path),
            "test_size": test_size,
            "model": model,
        },
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "f1_macro": metrics.get("f1_macro"),
            "confusion_matrix": metrics.get("confusion_matrix", []).tolist()
            if metrics
            else [],
            "classification_report": metrics.get("report", ""),
        },
        "classes": classes,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved metrics to {metrics_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)

    n_correct = sum(1 for c in classifications if c["correct"])
    n_wrong = len(classifications) - n_correct
    logger.info(f"Correct: {n_correct}, Wrong: {n_wrong}, Errors: {len(errors)}")
    logger.info(
        f"Total tokens used: {total_tokens.total_tokens:,} "
        f"(prompt: {total_tokens.prompt_tokens:,}, completion: {total_tokens.completion_tokens:,})"
    )

    if metrics:
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Macro: {metrics['f1_macro']:.4f}")

        # Log confusion matrix
        cm = metrics["confusion_matrix"]
        logger.info("Confusion Matrix:")
        # Header
        header = "            " + "  ".join(f"{c[:8]:>8}" for c in classes)
        logger.info(header)
        # Rows
        for i, row in enumerate(cm):
            row_str = f"{classes[i][:10]:>10}  " + "  ".join(f"{v:>8}" for v in row)
            logger.info(row_str)

    logger.info(f"Results saved to: {output_path}")

    return {
        "metrics": metrics,
        "classifications_path": str(classifications_path),
        "metrics_path": str(metrics_path),
        "log_path": str(output_path / "run.log"),
        "total": len(test_df),
        "classified": len(classifications),
        "correct": n_correct,
        "wrong": n_wrong,
        "errors": len(errors),
        "token_usage": {
            "prompt_tokens": total_tokens.prompt_tokens,
            "completion_tokens": total_tokens.completion_tokens,
            "total_tokens": total_tokens.total_tokens,
        },
    }
