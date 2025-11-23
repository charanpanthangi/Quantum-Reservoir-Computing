"""SVG-only plotting helpers for the QRC demo.

Matplotlib is configured to always save SVG files so the repository stays
friendly to version control and GitHub previews.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")


def _save_svg(fig: plt.Figure, output_path: str) -> None:
    """Save a Matplotlib figure as SVG and close it."""
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_reservoir_states(reservoir_states: np.ndarray, output_path: str) -> None:
    """Plot reservoir activations over time as a heatmap."""
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(reservoir_states, cmap="mako", ax=ax, cbar_kws={"label": "Activation"})
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Time step")
    ax.set_title("Quantum reservoir states")
    _save_svg(fig, output_path)


def plot_prediction_vs_target(y_true: np.ndarray, y_qrc: np.ndarray, y_classical: np.ndarray, output_path: str) -> None:
    """Overlay target values and predictions for visual comparison."""
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(y_true, label="Target", color="black", linewidth=2)
    ax.plot(y_qrc, label="Quantum RC", linestyle="--")
    ax.plot(y_classical, label="Classical RC", linestyle=":")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.set_title("Next-step prediction")
    ax.legend()
    _save_svg(fig, output_path)


def plot_mse_comparison(mse_qrc: float, mse_classical: float, output_path: str) -> None:
    """Bar chart comparing the mean squared error of both models."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["Quantum RC", "Classical RC"], [mse_qrc, mse_classical], color=["#6a5acd", "#20b2aa"])
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Prediction error")
    _save_svg(fig, output_path)
