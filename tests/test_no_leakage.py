"""Regression tests for the data leak this project exists to fix.

The original notebook computed encoding statistics over train + validation +
competition data concatenated together. That inflates every score, and it is
invisible: the code runs, the metrics just quietly lie. These tests make the
leak *fail loudly* if anyone reintroduces it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.features import FeatureEngineer

from .conftest import make_raw


def test_transform_is_row_independent(raw_train, raw_val):
    """The marquee test: transforming a row must not depend on its neighbours.

    If any statistic were computed from the frame being transformed (the
    definition of the leak), then padding that frame with extra rows would
    change the output for the original rows. It must not.
    """
    fe = FeatureEngineer().fit(raw_train)

    alone = fe.transform(raw_val)
    padded = fe.transform(pd.concat([raw_val, make_raw(n=300, seed=7)],
                                    ignore_index=True))

    pd.testing.assert_frame_equal(alone, padded.iloc[:len(raw_val)],
                                  check_dtype=False)


def test_unseen_category_gets_zero_frequency(raw_train):
    """A category that appears only at transform time was never counted at fit.

    Its frequency encoding must be 0. Anything else means validation rows
    contributed to the frequency map.
    """
    fe = FeatureEngineer().fit(raw_train)

    unseen = make_raw(n=50, seed=3, ward_levels=("ward_never_seen_in_train",))
    out = fe.transform(unseen)

    assert (out["ward_freq"] == 0).all()


def test_fitted_state_ignores_later_data(raw_train, raw_val):
    """Calling transform must never mutate the learned encoders."""
    fe = FeatureEngineer().fit(raw_train)
    before = {
        "medians": dict(fe.medians_),
        "district_mode": fe.district_mode_,
        "freq": {k: dict(v) for k, v in fe.freq_maps_.items()},
        "columns": list(fe.onehot_columns_),
    }

    fe.transform(raw_val)
    fe.transform(make_raw(n=200, seed=11))

    assert fe.medians_ == before["medians"]
    assert fe.district_mode_ == before["district_mode"]
    assert {k: dict(v) for k, v in fe.freq_maps_.items()} == before["freq"]
    assert list(fe.onehot_columns_) == before["columns"]


def test_imputation_values_come_from_train_only(raw_train):
    """The median used to fill zeros is the train median, ignoring zeros."""
    fe = FeatureEngineer().fit(raw_train)

    expected = raw_train["construction_year"].replace(0, np.nan).median()
    assert fe.medians_["construction_year"] == expected

    # And a frame of *only* zeros must be filled with that same train value.
    zeros = make_raw(n=30, seed=5)
    zeros["construction_year"] = 0
    out = fe.transform(zeros)
    assert (out["construction_year"] == expected).all()


def test_column_layout_is_locked_to_train(raw_train):
    """Categories unseen in train must not add columns; missing ones stay at 0."""
    fe = FeatureEngineer().fit(raw_train)

    odd = make_raw(n=40, seed=13)
    odd["quantity"] = "a_brand_new_level"        # never seen during fit
    out = fe.transform(odd)

    assert list(out.columns) == list(fe.onehot_columns_)
    assert (out["quantity_dry"] == 0).all()
