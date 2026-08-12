"""Generate the figures embedded in the README.

Run with::

    python -m src.figures

Every figure is written to ``reports/figures/`` as a PNG and committed, so the
README renders on GitHub without anyone having to execute a notebook.

**On the palette.** The three pump states look like a natural fit for a
good/warning/critical status palette, but a green-vs-red pair is the classic
colour-vision trap: simulated deuteranopia puts those two at OKLab ΔE 4.1,
below the ΔE 6 floor, and this data is drawn as a *scatter map* where every
pair of classes touches. The palette below was chosen by running the pairwise
CVD check instead of by eye — blue/yellow/red separates at ΔE 15.3 under
deuteranopia while keeping the intuitive reading (red = broken, yellow = at
risk). Yellow sits below 3:1 against the surface, so every figure using it
carries direct labels or a legend rather than relying on colour alone.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless: works in CI with no display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C
from . import data as D
from . import decision as DEC
from . import model as M
from .features import FeatureEngineer

# --- Design tokens -----------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = "#2a78d6"          # single-series bars (categorical slot 1)
STATUS_COLORS = {
    "functional": "#2a78d6",
    "functional needs repair": "#eda100",
    "non functional": "#e34948",
}
# Blue sequential ramp, light -> dark, for magnitude encoding.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

ORDER = ["functional", "non functional", "functional needs repair"]
SHORT = {"functional": "functional",
         "non functional": "non functional",
         "functional needs repair": "needs repair"}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.titlecolor": INK,
        "axes.titlesize": 15,
        "axes.titleweight": "600",
        "axes.labelsize": 11,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })


def _save(fig, name: str) -> None:
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(C.ROOT)}")


# --- Figures -----------------------------------------------------------------
def fig_target_distribution(df: pd.DataFrame) -> None:
    counts = df["status_group"].value_counts().reindex(ORDER)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(counts)), counts.values, width=0.62,
           color=[STATUS_COLORS[c] for c in counts.index])
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([SHORT[c] for c in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + counts.max() * 0.02, f"{v:,}  ({v / len(df) * 100:.0f}%)",
                ha="center", va="bottom", fontsize=11, color=INK)
    needs_crew = df["status_group"].isin(
        ["non functional", "functional needs repair"]).mean()
    ax.set_ylabel("Pumps")
    ax.set_title(f"{needs_crew:.0%} of pumps are broken or failing")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    _save(fig, "target_distribution.png")


def fig_zeros_as_missing(df: pd.DataFrame) -> None:
    cols = ["construction_year", "longitude", "gps_height", "population", "amount_tsh"]
    pct = pd.Series({c: (df[c] == 0).mean() * 100 for c in cols}).sort_values()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(range(len(pct)), pct.values, height=0.6, color=SERIES)
    ax.set_yticks(range(len(pct)))
    ax.set_yticklabels(pct.index)
    for i, v in enumerate(pct.values):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=11, color=INK)
    ax.set_xlabel("% of rows stored as 0, where 0 is physically impossible")
    ax.set_title("Missing data hiding as zeros")
    ax.set_xlim(0, max(pct.values) * 1.18)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", length=0)
    _save(fig, "zeros_as_missing.png")


def fig_map(df: pd.DataFrame) -> None:
    geo = df[(df["longitude"] > 25) & (df["latitude"] < -0.5)]  # drop invalid (0,0)

    fig, ax = plt.subplots(figsize=(8, 8))
    # Draw the majority class first so the rare "needs repair" stays visible.
    for status in ["functional", "non functional", "functional needs repair"]:
        s = geo[geo["status_group"] == status]
        ax.scatter(s["longitude"], s["latitude"], s=4, alpha=0.30, linewidths=0,
                   color=STATUS_COLORS[status], label=f"{SHORT[status]}  ({len(s):,})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Failure clusters geographically — location is the strongest signal")
    leg = ax.legend(markerscale=4, fontsize=10, loc="lower left", frameon=True,
                    facecolor=SURFACE, edgecolor=GRID)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    ax.set_aspect("equal", adjustable="datalim")
    _save(fig, "pump_map.png")


def fig_confusion_matrix(y_val, y_pred, model_name: str) -> None:
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.metrics import confusion_matrix

    labels = [1, 2, 0]  # functional, needs repair, non functional
    names = [SHORT[C.CODE_TO_LABEL[i]] for i in labels]
    cm = confusion_matrix(y_val, y_pred, labels=labels)
    pct = cm / cm.sum(axis=1, keepdims=True)

    cmap = LinearSegmentedColormap.from_list("blues", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    # pcolormesh (not imshow) so the cells can carry a 2px surface gap.
    ax.pcolormesh(pct, cmap=cmap, vmin=0, vmax=1,
                  edgecolors=SURFACE, linewidth=2)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j + 0.5, i + 0.5, f"{cm[i, j]:,}\n{pct[i, j]:.0%}",
                    ha="center", va="center", fontsize=11,
                    color="#ffffff" if pct[i, j] > 0.5 else INK)

    ax.set_xticks(np.arange(len(labels)) + 0.5, names)
    ax.set_yticks(np.arange(len(labels)) + 0.5, names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} — where it gets confused")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    _save(fig, "confusion_matrix.png")


def fig_feature_importance(model, feature_names, model_name: str) -> None:
    imp = M.feature_importance(model, feature_names, top=12).sort_values()
    imp = imp / imp.max()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(range(len(imp)), imp.values, height=0.62, color=SERIES)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.index)
    ax.set_xlabel("Relative importance")
    ax.set_title(f"What predicts pump failure ({model_name})")
    ax.set_xlim(0, 1.06)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", length=0)
    _save(fig, "feature_importance.png")


def fig_precision_at_k(y_val, proba) -> None:
    """The business chart: how well a limited inspection budget is spent."""
    scores = DEC.risk_score(proba)
    order = np.argsort(scores)[::-1]
    hits = np.isin(np.asarray(y_val)[order], DEC.NEEDS_ATTENTION)
    ks = np.arange(1, len(hits) + 1)
    precision = np.cumsum(hits) / ks
    base = hits.mean()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(ks, precision * 100, color=SERIES, linewidth=2, label="Model-ranked inspections")
    ax.axhline(base * 100, color=MUTED, linewidth=2, label="Random inspections (baseline)")

    ax.text(len(ks) * 0.60, base * 100 + 3, f"random: {base * 100:.0f}%",
            fontsize=10, color=INK_2)
    # Label below the curve: above it the annotations collide with the title.
    for k in (1000, 3000):
        if k < len(precision):
            p = precision[k - 1] * 100
            ax.plot([k], [p], "o", color=SERIES, markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2)
            ax.annotate(f"top {k:,}: {p:.0f}%", (k, p), textcoords="offset points",
                        xytext=(14, -22), fontsize=10, color=INK)

    ax.set_xlabel("Pumps inspected, ranked by predicted risk (k)")
    ax.set_ylabel("% that genuinely need a crew")
    ax.set_title("Ranking beats random: more broken pumps found per crew visit")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, len(ks))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, frameon=True, facecolor=SURFACE, edgecolor=GRID,
              loc="lower left")
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    _save(fig, "precision_at_k.png")


def main() -> None:
    _style()
    print("Loading data...")
    df = D.load_training()

    print("Rendering EDA figures...")
    fig_target_distribution(df)
    fig_zeros_as_missing(df)
    fig_map(df)

    print("Training models for the result figures...")
    y = D.encode_target(df[C.TARGET])
    X = df.drop(columns=[C.TARGET])
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE, stratify=y)
    fe = FeatureEngineer()
    X_tr = fe.fit_transform(X_tr_raw)
    X_val = fe.transform(X_val_raw)

    rf = M.train_random_forest(X_tr, y_tr)
    lgbm = M.train_lightgbm(X_tr, y_tr, X_val, y_val)
    from sklearn.metrics import f1_score
    candidates = {
        "Random Forest": rf,
        "LightGBM": lgbm,
    }
    name, best = max(candidates.items(),
                     key=lambda kv: f1_score(y_val, kv[1].predict(X_val), average="macro"))
    print(f"Best model: {name}")

    print("Rendering result figures...")
    fig_confusion_matrix(y_val, best.predict(X_val), name)
    fig_feature_importance(best, X_tr.columns, name)
    fig_precision_at_k(y_val, best.predict_proba(X_val))
    print("Done.")


if __name__ == "__main__":
    main()
