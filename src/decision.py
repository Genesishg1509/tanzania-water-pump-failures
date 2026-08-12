"""Turning probabilities into repair decisions.

A classifier that reports ``macro F1 = 0.70`` says nothing about whether it is
useful. The people who would actually run this model do not choose a *label* —
they choose **which pumps to send a crew to**, with a fixed number of crews and
a fixed budget. This module bridges that gap in two ways.

**1. Risk ranking (``precision_at_k`` / ``lift_at_k``).**
Field capacity is limited, so the real question is: *if we can only inspect k
pumps this quarter, how many broken ones do we find?* Ranking by predicted risk
and measuring precision@k answers that directly, and lift@k states it as a
multiple of the do-nothing baseline (inspecting k pumps at random).

**2. Cost-sensitive decisions (``expected_cost_predict``).**
``argmax P(class)`` implicitly assumes every mistake costs the same. It does
not: leaving a genuinely broken pump unvisited strands a village for months,
while a wasted inspection costs one crew-day. Given a cost matrix we can
instead pick the action that minimises *expected cost*, which trades cheap
false alarms for expensive misses.

Class codes follow ``config.LABEL_TO_CODE``: 0 = non functional,
1 = functional, 2 = functional needs repair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Classes that require dispatching a repair crew.
NEEDS_ATTENTION = (0, 2)

# --- Cost model --------------------------------------------------------------
# COST[true, predicted], expressed in units of "one wasted inspection visit".
#
# These are explicit, editable assumptions — not measured quantities. They
# encode three judgements a water authority would recognise:
#   * Missing a non-functional pump (true 0 -> predicted 1) is the worst
#     outcome: no crew is dispatched and a village stays without water until
#     the next survey cycle. Weighted 10x a wasted visit.
#   * Missing a pump that needs repair (true 2 -> predicted 1) is bad but less
#     acute — the pump still works today, though it will degrade. Weighted 5x.
#   * A false alarm (true 1 -> predicted 0 or 2) costs exactly one wasted visit.
#   * Confusing the two "needs a crew" classes with each other is nearly free:
#     the crew is dispatched either way and may simply carry the wrong parts.
DEFAULT_COST_MATRIX = np.array([
    # pred:  0 (non func)  1 (functional)  2 (needs repair)
    [0.0,  10.0,  1.0],   # true 0: non functional
    [1.0,   0.0,  1.0],   # true 1: functional
    [1.0,   5.0,  0.0],   # true 2: functional needs repair
])


def risk_score(proba: np.ndarray) -> np.ndarray:
    """P(pump needs a crew) = P(non functional) + P(needs repair).

    Collapsing the three class probabilities into one number is what makes the
    pumps rankable, which is the form a dispatcher can actually act on.
    """
    proba = np.asarray(proba)
    return proba[:, list(NEEDS_ATTENTION)].sum(axis=1)


def _attention_mask(y_true) -> np.ndarray:
    return np.isin(np.asarray(y_true), NEEDS_ATTENTION)


def precision_at_k(y_true, scores, k: int) -> float:
    """Share of the top-k riskiest pumps that genuinely need a crew."""
    k = min(k, len(scores))
    top = np.argsort(np.asarray(scores))[::-1][:k]
    return _attention_mask(y_true)[top].mean()


def recall_at_k(y_true, scores, k: int) -> float:
    """Share of all pumps needing a crew that are captured in the top k."""
    k = min(k, len(scores))
    top = np.argsort(np.asarray(scores))[::-1][:k]
    need = _attention_mask(y_true)
    return need[top].sum() / need.sum()


def lift_at_k(y_true, scores, k: int) -> float:
    """precision@k divided by the base rate — i.e. how much better than random.

    A lift of 1.0 means the model ranks no better than inspecting pumps at
    random, which is the honest baseline any field team already has.
    """
    base = _attention_mask(y_true).mean()
    return precision_at_k(y_true, scores, k) / base


def ranking_report(y_true, proba, ks=(500, 1000, 2500, 5000)) -> pd.DataFrame:
    """Precision / recall / lift at several inspection budgets."""
    scores = risk_score(proba)
    rows = [
        {
            "inspections (k)": k,
            "precision@k": precision_at_k(y_true, scores, k),
            "recall@k": recall_at_k(y_true, scores, k),
            "lift@k": lift_at_k(y_true, scores, k),
        }
        for k in ks
        if k <= len(scores)
    ]
    return pd.DataFrame(rows).set_index("inspections (k)")


def expected_cost_predict(proba: np.ndarray, cost_matrix=None) -> np.ndarray:
    """Predict the class with the lowest expected cost instead of the likeliest.

    For each pump the expected cost of choosing action ``j`` is
    ``sum_i P(true = i) * cost[i, j]``; we take the ``argmin``. When every
    mistake costs the same this reduces to ordinary ``argmax`` prediction.
    """
    cost_matrix = DEFAULT_COST_MATRIX if cost_matrix is None else np.asarray(cost_matrix)
    expected = np.asarray(proba) @ cost_matrix   # (n_samples, n_actions)
    return expected.argmin(axis=1)


def total_cost(y_true, y_pred, cost_matrix=None) -> float:
    """Total cost incurred by a set of decisions, in wasted-visit units."""
    cost_matrix = DEFAULT_COST_MATRIX if cost_matrix is None else np.asarray(cost_matrix)
    return float(cost_matrix[np.asarray(y_true), np.asarray(y_pred)].sum())


def cost_comparison(y_true, proba, cost_matrix=None) -> pd.DataFrame:
    """Compare accuracy-optimal vs cost-optimal decisions on the same model.

    Same probabilities, same model — only the decision rule changes. This
    isolates how much value the decision layer adds on its own.
    """
    cost_matrix = DEFAULT_COST_MATRIX if cost_matrix is None else np.asarray(cost_matrix)
    proba = np.asarray(proba)

    argmax_pred = proba.argmax(axis=1)
    cost_pred = expected_cost_predict(proba, cost_matrix)

    rows = []
    for name, pred in [("argmax P(class)", argmax_pred),
                       ("min expected cost", cost_pred)]:
        need = _attention_mask(y_true)
        dispatched = np.isin(pred, NEEDS_ATTENTION)
        rows.append({
            "decision rule": name,
            "accuracy": float((pred == np.asarray(y_true)).mean()),
            "missed broken pumps": int((need & ~dispatched).sum()),
            "wasted visits": int((~need & dispatched).sum()),
            "total cost": total_cost(y_true, pred, cost_matrix),
        })
    return pd.DataFrame(rows).set_index("decision rule")
