#!/usr/bin/env python
"""CLI interface for the IT Service Ticket Classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from classifier.artifacts import (
    ModelMetadata,
    load_model,
    load_rag_index,
    save_model,
    save_rag_index,
)
from classifier.classifiers import TfidfClassifier
from classifier.config import (
    EMBEDDING_MODEL,
    JUSTIFICATION,
    K_SIMILAR,
    RANDOM_STATE,
    REASONING_EFFORT,
    VALIDATION_SIZE,
)
from classifier.graph import GraphRuntime, build_eval_graph
from classifier.justifiers import LinearJustifier, LlmJustifier
from classifier.llm import LlmClient
from classifier.logging_config import setup_logging
from classifier.rag import TicketRetriever
from classifier.runner import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IT Service Ticket Classifier CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Train the model, evaluate classification, and save artifacts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument("dataset", type=str, help="Path to the CSV dataset file")
    train_parser.add_argument("--output", type=str, default="output", help="Directory for output files (logs, reports)")
    train_parser.add_argument(
        "--artifacts-dir", type=str, default="artifacts", help="Directory to save trained artifacts"
    )
    train_parser.add_argument(
        "--test-size",
        type=int,
        default=VALIDATION_SIZE,
        help="Number of validation samples (divided equally among classes)",
    )
    train_parser.add_argument(
        "--random-state", type=int, default=RANDOM_STATE, help="Random seed for reproducibility"
    )
    train_parser.add_argument(
        "--with-justifications",
        action="store_true",
        help="Generate justifications during evaluation",
    )
    train_parser.add_argument(
        "--justification",
        type=str,
        default=JUSTIFICATION,
        choices=["linear", "llm"],
        help="How to produce justifications when enabled",
    )
    train_parser.add_argument(
        "--k-similar",
        type=int,
        default=K_SIMILAR,
        help="Number of similar tickets to retrieve for context",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name (overrides LLM_MODEL env var)",
    )
    train_parser.add_argument(
        "--reasoning",
        type=str,
        default=REASONING_EFFORT,
        choices=["low", "medium", "high", "", None],
        help="Enable LLM reasoning with specified effort level",
    )
    train_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging to terminal",
    )

    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate a single ticket and return JSON output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    eval_parser.add_argument("ticket", nargs="?", help="Ticket text (omit to read from stdin)")
    eval_parser.add_argument(
        "--artifacts-dir", type=str, default="artifacts", help="Directory with trained artifacts"
    )
    eval_parser.add_argument(
        "--justification",
        type=str,
        default="linear",
        choices=["linear", "llm"],
        help="How to produce justifications",
    )
    eval_parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM-based justification",
    )
    eval_parser.add_argument(
        "--k-similar",
        type=int,
        default=K_SIMILAR,
        help="Number of similar tickets to retrieve for context",
    )
    eval_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name (overrides LLM_MODEL env var)",
    )
    eval_parser.add_argument(
        "--reasoning",
        type=str,
        default=REASONING_EFFORT,
        choices=["low", "medium", "high", "", None],
        help="Enable LLM reasoning with specified effort level",
    )
    eval_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging to terminal (stderr)",
    )

    return parser


def _load_ticket(ticket_arg: str | None) -> str:
    if ticket_arg is not None:
        return ticket_arg.strip()
    stdin_text = sys.stdin.read().strip()
    return stdin_text


def _build_justifier(
    classifier: TfidfClassifier,
    justification: str,
    *,
    model: str | None,
    reasoning_effort: str | None,
    k_similar: int,
    retriever: TicketRetriever | None,
) -> tuple[object, TicketRetriever | None]:
    if justification == "linear":
        justifier = LinearJustifier(
            classifier.classifier.coef_,
            classifier.classifier.classes_,
            classifier.features,
        )
        return justifier, None

    if retriever is None:
        raise ValueError("RAG index not found. Run train with LLM justifications.")

    llm_client = LlmClient(model=model, seed=None)
    justifier = LlmJustifier(
        llm_client,
        retriever,
        k_similar=k_similar,
        reasoning_effort=reasoning_effort,
    )
    return justifier, retriever


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        result = run_evaluation(
            dataset_path=args.dataset,
            output_dir=args.output,
            test_size=args.test_size,
            k_similar=args.k_similar,
            random_state=args.random_state,
            model=args.model,
            verbose=args.verbose,
            reasoning_effort=args.reasoning,
            justification=args.justification,
            with_justifications=args.with_justifications,
        )

        classifier = result["classifier"]
        classes = result["classes"]
        metadata = ModelMetadata(
            schema_version=1,
            random_state=args.random_state,
            classes=classes,
            model_name=classifier.name,
            embedding_model=EMBEDDING_MODEL,
        )
        save_model(
            classifier=classifier,
            metadata=metadata,
            artifacts_dir=args.artifacts_dir,
        )

        retriever = TicketRetriever()
        retriever.index(result["train_df"])
        save_rag_index(retriever, args.artifacts_dir)

        return 0

    if args.command == "eval":
        ticket = _load_ticket(args.ticket)
        if not ticket:
            print("Error: ticket text is required (arg or stdin).", file=sys.stderr)
            return 2

        if args.verbose:
            setup_logging(output_dir="output", verbose=True, console_stream=sys.stderr)

        model_path = Path(args.artifacts_dir) / "model.joblib"
        if not model_path.exists():
            print(
                "No trained model found. Run: uv run train dataset.csv",
                file=sys.stderr,
            )
            return 2
        classifier = load_model(args.artifacts_dir)

        justification_mode = "llm" if args.llm else args.justification
        retriever = None
        if justification_mode == "llm":
            rag_path = Path(args.artifacts_dir) / "rag"
            if not rag_path.exists():
                print(
                    "Error: RAG index not found. Run train with LLM justifications.",
                    file=sys.stderr,
                )
                return 2
            retriever = load_rag_index(args.artifacts_dir)

        justifier, retriever = _build_justifier(
            classifier=classifier,
            justification=justification_mode,
            model=args.model,
            reasoning_effort=args.reasoning,
            k_similar=args.k_similar,
            retriever=retriever,
        )

        runtime = GraphRuntime(
            classifier=classifier,
            justifier=justifier,
            justification_mode=justification_mode,
            retriever=retriever,
            k_similar=args.k_similar,
        )
        graph = build_eval_graph(runtime)
        result = graph.invoke({"ticket": ticket})
        output = result.get("output")
        if not output:
            print("Error: evaluation failed to produce output.", file=sys.stderr)
            return 1

        print(json.dumps(output, ensure_ascii=False))
        return 0

    parser.print_help()
    return 2


def train_cli() -> int:
    """Entry point for `uv run train`."""
    sys.argv = ["main.py", "train", *sys.argv[1:]]
    return main()


def eval_cli() -> int:
    """Entry point for `uv run eval`."""
    sys.argv = ["main.py", "eval", *sys.argv[1:]]
    return main()


if __name__ == "__main__":
    sys.exit(main())
