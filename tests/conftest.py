"""Shared fixtures.

The tests build a *synthetic* raw frame rather than reading ``data/raw``. The
real CSVs are not versioned (they belong to DrivenData), so tests that depended
on them could not run in CI — and a test suite that silently skips is worth
very little.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C

_CATEGORICAL_LEVELS = {
    "basin": ["lake victoria", "pangani", "rufiji"],
    "region": ["iringa", "mbeya", "kilimanjaro"],
    "scheme_management": ["vwc", "wug", "water board"],
    "extraction_type": ["gravity", "nira/tanira", "submersible"],
    "extraction_type_class": ["gravity", "handpump", "submersible"],
    "management": ["vwc", "wug", "private operator"],
    "payment": ["never pay", "pay annually", "pay per bucket"],
    "water_quality": ["soft", "salty", "milky"],
    "quantity": ["enough", "insufficient", "dry"],
    "source": ["spring", "shallow well", "river"],
    "source_class": ["groundwater", "surface"],
    "waterpoint_type": ["communal standpipe", "hand pump", "other"],
    # Dropped duplicates — present in the raw file, so present here too.
    "payment_type": ["never pay", "annually"],
    "quantity_group": ["enough", "dry"],
    "source_type": ["spring", "well"],
    "waterpoint_type_group": ["communal standpipe", "hand pump"],
    "extraction_type_group": ["gravity", "handpump"],
    "management_group": ["user-group", "commercial"],
    "quality_group": ["good", "salty"],
}


def make_raw(n: int = 400, seed: int = 0, ward_levels=("ward_a", "ward_b", "ward_c")):
    """Build a frame with the same schema as the DrivenData training file."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "id": np.arange(n),
        "amount_tsh": rng.choice([0.0, 50.0, 500.0, 1000.0], n),
        "date_recorded": rng.choice(["2011-03-14", "2013-02-04", "2012-10-01"], n),
        "funder": rng.choice(["gov", "danida", "world bank", None], n),
        "gps_height": rng.integers(0, 2000, n).astype(float),
        "installer": rng.choice(["dwe", "gov", "commu", None], n),
        # Keep some exact zeros: the imputation path for "0 means missing".
        "longitude": rng.choice([0.0, 34.5, 37.1, 39.2], n),
        "latitude": rng.uniform(-11.0, -1.0, n),
        "wpt_name": [f"point_{i}" for i in range(n)],
        "num_private": np.zeros(n),
        "subvillage": rng.choice([f"sv_{i}" for i in range(40)] + [None], n),
        "region_code": rng.integers(1, 20, n),
        "district_code": rng.choice([0, 1, 2, 3, 4], n),
        "lga": rng.choice(["lga_a", "lga_b", "lga_c"], n),
        "ward": rng.choice(list(ward_levels), n),
        "population": rng.integers(0, 500, n).astype(float),
        "public_meeting": rng.choice([True, False, None], n),
        "recorded_by": "GeoData Consultants Ltd",
        "scheme_name": rng.choice(["scheme_x", "scheme_y", None], n),
        "permit": rng.choice([True, False, None], n),
        "construction_year": rng.choice([0, 1995, 2003, 2009], n),
    })
    for col, levels in _CATEGORICAL_LEVELS.items():
        df[col] = rng.choice(levels, n)

    df[C.TARGET] = rng.choice(list(C.LABEL_TO_CODE), n)
    return df


@pytest.fixture
def raw_train():
    return make_raw(n=400, seed=0)


@pytest.fixture
def raw_val():
    return make_raw(n=120, seed=99)
