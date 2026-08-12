"""Tests for the risk-ranking and cost-sensitive decision layer."""
from __future__ import annotations

import numpy as np
import pytest

from src import decision as DEC


@pytest.fixture
def proba():
    """Rows 0-2 confidently non functional / functional / needs repair."""
    return np.array([
        [0.90, 0.05, 0.05],   # non functional
        [0.05, 0.90, 0.05],   # functional
        [0.05, 0.05, 0.90],   # needs repair
        [0.40, 0.35, 0.25],   # uncertain, leans broken
    ])


def test_risk_score_sums_the_two_actionable_classes(proba):
    expected = proba[:, 0] + proba[:, 2]
    np.testing.assert_allclose(DEC.risk_score(proba), expected)


def test_risk_score_is_a_probability():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([1, 1, 1], size=200)
    scores = DEC.risk_score(p)
    assert np.all((scores >= 0) & (scores <= 1))


def test_perfect_ranking_gives_precision_one():
    # Two pumps need a crew (codes 0 and 2) and they carry the highest scores.
    y = np.array([0, 2, 1, 1])
    scores = np.array([0.99, 0.98, 0.02, 0.01])
    assert DEC.precision_at_k(y, scores, 2) == 1.0


def test_recall_at_full_budget_is_one():
    y = np.array([0, 2, 1, 1, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.7])
    assert DEC.recall_at_k(y, scores, len(y)) == 1.0


def test_lift_at_full_budget_is_one():
    """Inspecting everything cannot beat random — lift must collapse to 1."""
    y = np.array([0, 2, 1, 1, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.7])
    assert DEC.lift_at_k(y, scores, len(y)) == pytest.approx(1.0)


def test_k_larger_than_the_dataset_is_clamped():
    y = np.array([0, 1, 1])
    scores = np.array([0.9, 0.2, 0.1])
    assert DEC.precision_at_k(y, scores, 999) == DEC.precision_at_k(y, scores, 3)


def test_uniform_costs_reduce_to_argmax(proba):
    """With every mistake priced the same, the cost rule *is* the argmax rule."""
    uniform = 1.0 - np.eye(3)
    np.testing.assert_array_equal(
        DEC.expected_cost_predict(proba, uniform), proba.argmax(axis=1))


def test_cost_rule_never_loses_to_argmax_on_cost():
    """The whole point of the decision layer, stated as a property.

    Minimising expected cost cannot do worse *on cost* than maximising
    probability — if it ever does, the implementation is wrong.
    """
    rng = np.random.default_rng(42)
    p = rng.dirichlet([1.5, 3.0, 0.6], size=4000)
    y = np.array([rng.choice(3, p=row) for row in p])

    argmax_cost = DEC.total_cost(y, p.argmax(axis=1))
    cost_optimal = DEC.total_cost(y, DEC.expected_cost_predict(p))

    assert cost_optimal <= argmax_cost


def test_cost_rule_dispatches_more_crews_than_argmax():
    """Because misses are priced above false alarms, it should err toward visiting."""
    rng = np.random.default_rng(7)
    p = rng.dirichlet([1.5, 3.0, 0.6], size=2000)

    argmax_dispatch = np.isin(p.argmax(axis=1), DEC.NEEDS_ATTENTION).sum()
    cost_dispatch = np.isin(DEC.expected_cost_predict(p), DEC.NEEDS_ATTENTION).sum()

    assert cost_dispatch >= argmax_dispatch


def test_perfect_predictions_cost_nothing():
    y = np.array([0, 1, 2, 1, 0])
    assert DEC.total_cost(y, y) == 0.0


def test_cost_matrix_has_a_zero_diagonal():
    """A correct decision must be free, or every downstream total is offset."""
    np.testing.assert_array_equal(np.diag(DEC.DEFAULT_COST_MATRIX), np.zeros(3))


def test_missing_a_broken_pump_costs_more_than_a_wasted_visit():
    """The asymmetry the whole module exists to express."""
    cost = DEC.DEFAULT_COST_MATRIX
    miss_broken = cost[0, 1]      # truly non functional, predicted functional
    false_alarm = cost[1, 0]      # truly functional, crew sent anyway
    assert miss_broken > false_alarm


def test_ranking_report_shape_and_monotonicity():
    rng = np.random.default_rng(1)
    p = rng.dirichlet([1.5, 3.0, 0.6], size=3000)
    y = np.array([rng.choice(3, p=row) for row in p])

    report = DEC.ranking_report(y, p, ks=(100, 500, 1000))
    assert list(report.index) == [100, 500, 1000]
    # Recall can only grow as the budget grows.
    assert report["recall@k"].is_monotonic_increasing


def test_ranking_report_skips_budgets_larger_than_the_data():
    rng = np.random.default_rng(2)
    p = rng.dirichlet([1, 1, 1], size=50)
    y = rng.integers(0, 3, size=50)
    report = DEC.ranking_report(y, p, ks=(10, 1000))
    assert list(report.index) == [10]
