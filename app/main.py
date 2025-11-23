"""Command-line entry point for the QRC demo.

Steps performed:
1. Generate a noisy sine wave dataset.
2. Build supervised windows for next-step prediction.
3. Train quantum and classical reservoirs with a linear readout.
4. Save SVG plots comparing results.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import dataset
from .plots import plot_mse_comparison, plot_prediction_vs_target, plot_reservoir_states
from .trainer import train_qrc_and_baseline
from .utils import standardize, destandardize


def parse_args() -> argparse.Namespace:
    """Parse simple CLI arguments with sensible defaults."""
    parser = argparse.ArgumentParser(description="Quantum Reservoir Computing demo")
    parser.add_argument("--n-points", type=int, default=300, help="Number of points in the synthetic sequence")
    parser.add_argument("--window-size", type=int, default=5, help="Sliding window size for inputs")
    parser.add_argument("--n-qubits", type=int, default=2, help="Number of qubits in the quantum reservoir")
    parser.add_argument("--depth", type=int, default=2, help="Depth of the random quantum circuit")
    parser.add_argument("--classical-dim", type=int, default=50, help="Size of the classical reservoir")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level for the sine wave")
    parser.add_argument("--freq", type=float, default=1.0, help="Frequency multiplier for the sine wave")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)

    # 1) Build dataset
    sequence = dataset.generate_sine_sequence(n_points=args.n_points, noise=args.noise, freq=args.freq, seed=args.seed)
    X, y = dataset.build_supervised_pairs(sequence, window_size=args.window_size)
    X_train, X_test, y_train, y_test = dataset.train_test_split(X, y, seed=args.seed)

    # Standardize targets for stable training and revert later
    y_train_std, mean, std = standardize(y_train)
    y_test_std = (y_test - mean) / (std + 1e-8)

    # 2) Train models
    results = train_qrc_and_baseline(
        X_train,
        y_train_std,
        X_test,
        y_test_std,
        n_qubits=args.n_qubits,
        depth=args.depth,
        classical_dim=args.classical_dim,
        seed=args.seed,
    )

    # Convert predictions back to original scale
    y_q_pred = destandardize(results["pred_qrc"], mean, std)
    y_c_pred = destandardize(results["pred_classical"], mean, std)

    mse_q = ((y_test - y_q_pred) ** 2).mean()
    mse_c = ((y_test - y_c_pred) ** 2).mean()

    print("Quantum Reservoir MSE:", mse_q)
    print("Classical Reservoir MSE:", mse_c)

    # 3) Save SVG plots
    plot_reservoir_states(results["reservoir_states_qrc"], examples_dir / "qrc_reservoir_states.svg")
    plot_prediction_vs_target(y_test, y_q_pred, y_c_pred, examples_dir / "qrc_prediction_vs_target.svg")
    plot_mse_comparison(mse_q, mse_c, examples_dir / "qrc_vs_classical_reservoir.svg")

    print("SVG plots saved to:")
    for path in [
        examples_dir / "qrc_reservoir_states.svg",
        examples_dir / "qrc_prediction_vs_target.svg",
        examples_dir / "qrc_vs_classical_reservoir.svg",
    ]:
        print(" -", path)


if __name__ == "__main__":
    main()
