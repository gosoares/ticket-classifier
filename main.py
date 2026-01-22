#!/usr/bin/env python
"""CLI interface for the IT Service Ticket Classifier."""

import argparse
import sys

from classifier.config import (
    JUSTIFICATION,
    K_SIMILAR,
    RANDOM_STATE,
    REASONING_EFFORT,
    VALIDATION_SIZE,
)
from classifier.runner import run_evaluation


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="IT Service Ticket Classifier - Evaluate ML classification + justifications (linear by default, optional LLM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset.csv",
        help="Path to the CSV dataset file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Directory for output files (logs, reports)",
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=VALIDATION_SIZE,
        help="Number of validation samples (divided equally among classes)",
    )

    parser.add_argument(
        "--k-similar",
        type=int,
        default=K_SIMILAR,
        help="Number of similar tickets to retrieve for context",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name (overrides LLM_MODEL env var)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging to terminal",
    )

    parser.add_argument(
        "--reasoning",
        type=str,
        default=REASONING_EFFORT,
        choices=["low", "medium", "high", "", None],
        help="Enable LLM reasoning with specified effort level",
    )

    parser.add_argument(
        "--justification",
        type=str,
        default=JUSTIFICATION,
        choices=["linear", "llm"],
        help="How to produce justifications (linear is faster and avoids LLM dependency)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the CLI."""
    args = parse_args()

    try:
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
        )

        # Print summary to stdout
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY (VALIDATION SET)")
        print("=" * 60)
        print(f"Total tickets:  {result['total']}")
        print(f"Correct:        {result['correct']}")
        print(f"Wrong:          {result['wrong']}")
        print(f"Errors:         {result['errors']}")

        if result["metrics"]:
            print(f"\nAccuracy:       {result['metrics']['accuracy']:.4f}")
            print(f"F1 Macro:       {result['metrics']['f1_macro']:.4f}")

        if result.get("token_usage"):
            tokens = result["token_usage"]
            print("\nToken usage:")
            print(f"  Prompt:       {tokens['prompt_tokens']:,}")
            print(f"  Completion:   {tokens['completion_tokens']:,}")
            print(f"  Total:        {tokens['total_tokens']:,}")

        if result.get("conclusion"):
            print("\n" + "=" * 60)
            print("CONCLUSÃO (LLM)")
            print("=" * 60)
            print(result["conclusion"])

        print("\nOutput files:")
        print(f"  - {result['classifications_path']}")
        print(f"  - {result['metrics_path']}")
        print(f"  - {result['log_path']}")
        if result.get("conclusion_path"):
            print(f"  - {result['conclusion_path']}")

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
