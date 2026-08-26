"""Run the learning-rate tuning toy experiment from Assignment 1.

The update is the SGD variant shown in the handout:

    theta[t + 1] = theta[t] - learning_rate / sqrt(t + 1) * grad

Usage:
    .venv/bin/python experiments/learning_rate_tuning.py
"""

import argparse
import json
import math
from pathlib import Path

import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EXPERIMENTS_DIR / "artifacts" / "learning_rate_tuning.json"
DEFAULT_LEARNING_RATES = (1.0, 10.0, 100.0, 1000.0)


def run_sgd(initial_weights: torch.Tensor, learning_rate: float, iterations: int) -> list[float]:
    """Return the loss before each of ``iterations`` handout-style updates."""
    weights = initial_weights.clone().requires_grad_(True)
    losses: list[float] = []

    for iteration in range(iterations):
        loss = (weights**2).mean()
        losses.append(loss.item())
        loss.backward()

        # The handout's optimizer divides the step size by sqrt(t + 1).
        with torch.no_grad():
            weights -= learning_rate / math.sqrt(iteration + 1) * weights.grad
        weights.grad = None

    return losses


def classify_run(losses: list[float]) -> str:
    """Classify a run using its final loss relative to its initial loss."""
    if not losses:
        return "no iterations"
    if losses[-1] > losses[0]:
        return "diverges"
    if losses[-1] < losses[0] * 0.5:
        return "fast decay"
    return "slow decay"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=list(DEFAULT_LEARNING_RATES),
        metavar="LR",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if any(rate <= 0 for rate in args.learning_rates):
        raise ValueError("learning rates must be positive")

    torch.manual_seed(args.seed)
    initial_weights = 5 * torch.randn((10, 10))
    results = []
    for learning_rate in args.learning_rates:
        losses = run_sgd(initial_weights, learning_rate, args.iterations)
        results.append(
            {
                "learning_rate": learning_rate,
                "losses": losses,
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "behavior": classify_run(losses),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(
            {"seed": args.seed, "iterations": args.iterations, "results": results},
            file,
            indent=2,
        )

    print("| learning rate | initial loss | final loss | behavior |")
    print("|---:|---:|---:|---|")
    for result in results:
        print(
            f"| {result['learning_rate']:g} | {result['initial_loss']:.6g} "
            f"| {result['final_loss']:.6g} | {result['behavior']} |"
        )
    print(f"saved results: {args.output}")


if __name__ == "__main__":
    main()
