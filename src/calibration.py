"""Probability calibration and diagnostics.

decision.py::expected_cost_predict trusts the model's raw probabilities to
minimise expected cost. If those probabilities are miscalibrated — the model
says "70% risk" but only 50% of such pumps are actually at-risk — the
cost-minimising decision is wrong even when the ranking (precision@k) looks
fine. This module calibrates the winning model on a held-out split and
measures whether it worked.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless: works in CI with no display

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

# Same colour-vision-checked palette as figures.py (see its docstring).
CLASS_COLORS = {0: "#e34948", 1: "#2a78d6", 2: "#eda100"}
CLASS_NAMES = {0: "non functional", 1: "functional", 2: "needs repair"}


def calibrate(model, X_calib, y_calib, method: str = "sigmoid") -> CalibratedClassifierCV:
    """Calibrate an already-fitted model using a held-out split.

    FrozenEstimator tells CalibratedClassifierCV not to refit ``model`` — only
    the calibration mapping is learned, on data the model never trained on.
    (sklearn dropped ``cv="prefit"`` in 1.6; this is its replacement.)
    """
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    calibrated.fit(X_calib, y_calib)
    return calibrated


def expected_calibration_error(y_binary, proba_class, n_bins: int = 10) -> float:
    """Bin-size-weighted mean |confidence - accuracy| across probability bins."""
    proba_class = np.asarray(proba_class)
    y_binary = np.asarray(y_binary)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(proba_class, bins[1:-1])

    ece, n = 0.0, len(proba_class)
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        confidence = proba_class[mask].mean()
        accuracy = y_binary[mask].mean()
        ece += (mask.sum() / n) * abs(confidence - accuracy)
    return float(ece)


def calibration_metrics(y_true, proba, labels=(0, 1, 2)) -> dict:
    """Brier score and ECE per class, one-vs-rest."""
    y_true = np.asarray(y_true)
    out = {}
    for i, label in enumerate(labels):
        y_binary = (y_true == label).astype(int)
        p = proba[:, i]
        out[CLASS_NAMES[label]] = {
            "brier_score": float(brier_score_loss(y_binary, p)),
            "ece": expected_calibration_error(y_binary, p),
        }
    return out


def reliability_diagram(y_true, proba, labels=(0, 1, 2), n_bins: int = 10):
    """One-vs-rest reliability diagram, one subplot per class."""
    y_true = np.asarray(y_true)
    fig, axes = plt.subplots(1, len(labels), figsize=(4.5 * len(labels), 4.5))

    for i, (ax, label) in enumerate(zip(axes, labels)):
        y_binary = (y_true == label).astype(int)
        p = proba[:, i]
        frac_pos, mean_pred = calibration_curve(y_binary, p, n_bins=n_bins, strategy="uniform")

        color = CLASS_COLORS[label]
        ax.plot([0, 1], [0, 1], "--", color="#c3c2b7", linewidth=1.5, label="Perfectly calibrated")
        ax.plot(mean_pred, frac_pos, "o-", color=color, linewidth=2, markersize=6,
                label=CLASS_NAMES[label])
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(CLASS_NAMES[label])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9, loc="upper left", frameon=False)

    fig.suptitle("Reliability diagram — is predicted risk trustworthy?", fontsize=13)
    fig.tight_layout()
    return fig


def save_reliability_diagram(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
