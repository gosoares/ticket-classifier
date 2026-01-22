"""Runner module for executing the full evaluation pipeline."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from classifier.config import (
    EMBEDDING_MODEL,
    K_SIMILAR,
    RANDOM_STATE,
    REASONING_EFFORT,
    VALIDATION_SIZE,
)
from classifier.conclusion import (
    build_conclusion_payload,
    build_conclusion_system_prompt,
    build_conclusion_user_prompt,
)
from classifier.classifiers import TfidfClassifier
from classifier.data import load_dataset, train_test_validation_split
from classifier.graph import justify_ticket
from classifier.llm import (
    JustificationError,
    ConclusionError,
    TicketJustifier,
    TokenUsage,
)
from classifier.logging_config import get_logger, setup_logging
from classifier.metrics import evaluate
from classifier.rag import TicketRetriever

logger = get_logger("runner")


def classify_batch(
    test_df: pd.DataFrame,
    retriever: TicketRetriever,
    justifier: TicketJustifier,
    ml_classifier: TfidfClassifier,
    classes: list[str],
    k_similar: int = K_SIMILAR,
    reasoning_effort: str | None = REASONING_EFFORT,
    show_progress: bool = True,
) -> tuple[list[dict], list[dict], TokenUsage]:
    """
    Generate justifications for a batch using ML predictions.

    Args:
        test_df: DataFrame with 'Document' and 'Topic_group' columns
        retriever: Indexed TicketRetriever instance
        justifier: TicketJustifier instance
        ml_classifier: Trained TF-IDF classifier
        reasoning_effort: Reasoning effort level (low/medium/high) or None
        show_progress: Whether to show progress bar

    Returns:
        Tuple of (classifications, errors, total_tokens)
        - classifications: List of dicts with ticket, true_class, predicted_class,
          justification, reasoning, correct, system_prompt, user_prompt,
          similar_tickets, retries, token_usage
        - errors: List of dicts with ticket, true_class, reason, raw_response
        - total_tokens: TokenUsage with aggregated token counts
    """
    classifications = []
    errors = []
    y_true = []
    y_pred = []
    total_tokens = TokenUsage(0, 0, 0)

    pbar = tqdm(
        test_df.iterrows(),
        total=len(test_df),
        desc="Justifying",
        disable=not show_progress,
    )
    for _, row in pbar:
        ticket_text = row["Document"]
        true_label = row["Topic_group"]

        try:
            details = justify_ticket(
                ticket=ticket_text,
                retriever=retriever,
                justifier=justifier,
                ml_classifier=ml_classifier,
                reasoning_effort=reasoning_effort,
                k_similar=k_similar,
            )

            y_true.append(true_label)
            predicted_label = details.predicted_class
            y_pred.append(predicted_label)

            # Accumulate token usage
            total_tokens = total_tokens + details.token_usage

            # Update progress bar with last result
            is_correct = true_label == predicted_label
            if show_progress:
                pbar.set_postfix(pred=predicted_label, ok=is_correct)

            classifications.append(
                {
                    "ticket": ticket_text,
                    "true_class": true_label,
                    "predicted_class": predicted_label,
                    "justification": details.result.justificativa,
                    "reasoning": details.reasoning,
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

        except JustificationError as e:
            if show_progress:
                pbar.set_postfix(error=e.reason[:20])
            errors.append(
                {
                    "ticket": ticket_text,
                    "true_class": true_label,
                    "reason": e.reason,
                    "raw_response": e.raw_response,
                }
            )

    return classifications, errors, total_tokens


def run_evaluation(
    dataset_path: str | Path = "dataset.csv",
    output_dir: str | Path = "output",
    test_size: int = VALIDATION_SIZE,
    k_similar: int = K_SIMILAR,
    random_state: int = RANDOM_STATE,
    model: str | None = None,
    verbose: bool = False,
    reasoning_effort: str | None = REASONING_EFFORT,
) -> dict:
    """
    Execute the full evaluation pipeline.

    Steps:
    1. Load dataset and split into train/test/validation
    2. Train ML classifier
    3. Index training tickets for RAG retrieval
    4. Generate justifications
    5. Calculate evaluation metrics
    6. Save results to output files

    Args:
        dataset_path: Path to the CSV dataset file
        output_dir: Directory for output files
        test_size: Number of validation samples (divided equally among classes)
        k_similar: Number of similar tickets to retrieve
        random_state: Random seed for reproducibility
        model: LLM model name (from LLM_MODEL env var if not specified)
        verbose: Enable verbose logging to terminal
        reasoning_effort: Reasoning effort level (low/medium/high) or None

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
    logger.info(f"Validation size: {test_size}")
    logger.info(f"K similar: {k_similar}")
    logger.info(f"Model: {model}")
    logger.info(f"Reasoning effort: {reasoning_effort}")
    logger.info(f"Random state: {random_state}")

    # Step 1: Load and split data
    logger.info("-" * 60)
    logger.info("Step 1: Loading and splitting dataset")
    df, classes = load_dataset(Path(dataset_path))
    train_df, test_df, validation_df = train_test_validation_split(
        df,
        validation_size=test_size,
        random_state=random_state,
    )

    # Step 2: Train ML classifier
    logger.info("-" * 60)
    logger.info("Step 2: Training ML classifier")
    ml_classifier = TfidfClassifier.linear_svc(random_state=random_state)
    ml_classifier.fit(train_df)
    logger.info("ML classifier trained")

    # Step 3: Index training data
    logger.info("-" * 60)
    logger.info("Step 3: Indexing training data for RAG")
    retriever = TicketRetriever()
    retriever.index(train_df)

    # Step 4: Generate justifications
    logger.info("-" * 60)
    logger.info("Step 4: Generating justifications with RAG + LLM (validation)")
    justifier = TicketJustifier(model=model, seed=random_state)

    classifications, errors, total_tokens = classify_batch(
        test_df=validation_df,
        retriever=retriever,
        justifier=justifier,
        ml_classifier=ml_classifier,
        classes=classes,
        k_similar=k_similar,
        reasoning_effort=reasoning_effort,
        show_progress=True,
    )

    # Extract y_true and y_pred from classifications for metrics
    y_true = [c["true_class"] for c in classifications]
    y_pred = [c["predicted_class"] for c in classifications]

    logger.info(f"Justified: {len(classifications)}, Errors: {len(errors)}")
    logger.info(
        f"Total tokens: {total_tokens.total_tokens} "
        f"(prompt: {total_tokens.prompt_tokens}, completion: {total_tokens.completion_tokens})"
    )

    # Step 5: Calculate metrics
    logger.info("-" * 60)
    logger.info("Step 5: Calculating metrics")
    metrics = {}
    if y_true:
        metrics = evaluate(y_true, y_pred, classes)
    else:
        logger.warning("No successful classifications, skipping metrics")

    # Step 5: Save outputs
    logger.info("-" * 60)
    logger.info("Step 6: Saving results")

    timestamp = datetime.now().isoformat()

    # Save classifications report
    classifications_path = output_path / "classifications.json"
    classifications_report = {
        "metadata": {
            "timestamp": timestamp,
            "dataset": str(dataset_path),
            "validation_size": test_size,
            "k_similar": k_similar,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "random_state": random_state,
        },
        "summary": {
            "total": len(validation_df),
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
    cm = metrics.get("confusion_matrix") if metrics else None
    cm_normalized = metrics.get("confusion_matrix_normalized") if metrics else None
    metrics_report = {
        "metadata": {
            "timestamp": timestamp,
            "dataset": str(dataset_path),
            "validation_size": test_size,
            "model": model,
        },
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "f1_macro": metrics.get("f1_macro"),
            "f1_weighted": metrics.get("f1_weighted"),
            "cohen_kappa": metrics.get("cohen_kappa"),
            "mcc": metrics.get("mcc"),
            "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else (cm or []),
            "confusion_matrix_normalized": cm_normalized.tolist()
            if hasattr(cm_normalized, "tolist")
            else (cm_normalized or []),
            "per_class": metrics.get("per_class", {}) if metrics else {},
            "classification_report": metrics.get("report", ""),
        },
        "classes": classes,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved metrics to {metrics_path}")

    # Step 6: Generate conclusion
    logger.info("-" * 60)
    logger.info("Step 7: Generating conclusion")
    conclusion_text = None
    conclusion_path = None
    conclusion_payload = None
    conclusion_token_usage = None
    try:
        conclusion_payload = build_conclusion_payload(
            dataset=str(dataset_path),
            classes=classes,
            test_size=len(validation_df),
            k_similar=k_similar,
            embedding_model=EMBEDDING_MODEL,
            llm_model=justifier.model,
            random_state=random_state,
            classifications=classifications,
            errors=errors,
            metrics=metrics,
            token_usage=total_tokens,
            max_misclassified=20,
            timestamp=timestamp,
        )
        system_prompt = build_conclusion_system_prompt()
        user_prompt = build_conclusion_user_prompt(conclusion_payload)
        conclusion_text, conclusion_token_usage = justifier.generate_conclusion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        conclusion_path = output_path / "conclusion.json"
        with open(conclusion_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "timestamp": timestamp,
                        "dataset": str(dataset_path),
                        "model": justifier.model,
                    },
                    "payload": conclusion_payload,
                    "conclusion": conclusion_text,
                    "token_usage": {
                        "prompt_tokens": conclusion_token_usage.prompt_tokens,
                        "completion_tokens": conclusion_token_usage.completion_tokens,
                        "total_tokens": conclusion_token_usage.total_tokens,
                    }
                    if conclusion_token_usage
                    else None,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"Saved conclusion to {conclusion_path}")
    except ConclusionError as e:
        logger.error(f"Conclusion generation failed: {e.reason}")

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
        logger.info(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
        logger.info(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        logger.info(f"MCC: {metrics['mcc']:.4f}")

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
        "conclusion": conclusion_text,
        "conclusion_path": str(conclusion_path) if conclusion_path else None,
        "conclusion_payload": conclusion_payload,
        "conclusion_token_usage": {
            "prompt_tokens": conclusion_token_usage.prompt_tokens,
            "completion_tokens": conclusion_token_usage.completion_tokens,
            "total_tokens": conclusion_token_usage.total_tokens,
        }
        if conclusion_token_usage
        else None,
        "total": len(validation_df),
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
