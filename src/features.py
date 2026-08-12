"""Feature engineering pipeline.

Design principle: **every statistic is learned from the training split only**
and then applied to validation / competition data. This avoids the data
leakage that inflates optimistic scores.

What changed vs. the original university notebook (and why):
  * Removed a ``pd.concat([train, test, comp])`` used to pick top-100 funders.
    That leaked competition data into training and — worse — its output was
    never actually used downstream (dead code).
  * Removed ``np.log1p`` on numeric features. Gradient-boosted / random-forest
    trees are invariant to monotonic transforms, so it added nothing; it was
    also being applied to ``latitude``/``longitude`` with ``.abs()`` (which
    corrupts Tanzania's all-negative latitudes) and to ``construction_year``.
  * Frequency encoding is fit on the training split only.
  * Feature-hashing buckets reduced to a sane size and kept nominal (never fed
    to SMOTE, which would interpolate meaningless fractional bucket ids).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


# --- Text normalization ------------------------------------------------------
def normalize_text(x):
    """Lowercase, trim, and collapse separators to ``_`` (NaNs pass through)."""
    if pd.isna(x):
        return x
    x = str(x).lower().strip()
    x = re.sub(r"\s+", "_", x)
    x = x.replace("/", "_").replace("-", "_")
    return x


def _hash_bucket(value: str, n_buckets: int) -> int:
    """Deterministic feature hashing of a single string into ``n_buckets``."""
    return int(hashlib.md5(str(value).encode()).hexdigest(), 16) % n_buckets


@dataclass
class FeatureEngineer:
    """Fit encoders/imputers on train, then transform any split identically."""

    # learned state (populated in ``fit``)
    medians_: dict = field(default_factory=dict)
    district_mode_: int = 0
    longitude_by_region_: pd.Series = None
    freq_maps_: dict = field(default_factory=dict)
    onehot_columns_: list = None

    # ---- fit ----------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        df = self._normalize(df.copy())

        # Zero-as-missing numeric columns -> train median (ignoring the zeros).
        for col in C.ZERO_AS_MISSING_MEDIAN:
            self.medians_[col] = df[col].replace(0, np.nan).median()

        # district_code: 0 is invalid -> train mode.
        self.district_mode_ = int(df["district_code"].replace(0, np.nan).mode()[0])

        # longitude: 0 is invalid -> conditional median by region (train only).
        valid = df[df["longitude"] != 0]
        self.longitude_by_region_ = valid.groupby("region")["longitude"].median()

        # Frequency maps (train only).
        for col in C.FREQ_COLS:
            self.freq_maps_[col] = df[col].value_counts().to_dict()

        # Record the one-hot column layout produced on train for alignment.
        transformed = self._transform_core(df)
        self.onehot_columns_ = transformed.columns.tolist()
        return self

    # ---- transform ----------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._transform_core(self._normalize(df.copy()))
        # Align to the training column layout (missing dummies -> 0, drop extras).
        out = out.reindex(columns=self.onehot_columns_, fill_value=0)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ---- internals ----------------------------------------------------------
    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        text_cols = df.select_dtypes(include=["object"]).columns
        for col in text_cols:
            df[col] = df[col].apply(normalize_text)
        return df

    def _transform_core(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1) Impute high-cardinality categoricals with "other".
        for col in C.FILL_OTHER:
            df[col] = df[col].fillna("other")

        # 2) Impute zero-as-missing numerics with the learned train median.
        for col in C.ZERO_AS_MISSING_MEDIAN:
            df[col] = df[col].replace(0, self.medians_[col])

        # 3) district_code zeros -> train mode.
        df["district_code"] = df["district_code"].replace(0, self.district_mode_)

        # 4) longitude zeros -> conditional median by region (fallback global).
        global_lon = self.longitude_by_region_.median()
        mask = df["longitude"] == 0
        df.loc[mask, "longitude"] = (
            df.loc[mask, "region"].map(self.longitude_by_region_).fillna(global_lon)
        )

        # 5) Date features.
        dr = df["date_recorded"].astype(str).str.replace("_", "-", regex=False)
        dr = pd.to_datetime(dr, errors="coerce")
        df["year_recorded"] = dr.dt.year
        df["years_in_operation"] = (
            df["year_recorded"] - df["construction_year"]
        ).clip(lower=0)  # negative ages are data errors -> floor at 0

        # 6) Feature hashing for very high cardinality columns.
        for col in C.HASH_COLS:
            df[col + "_hashed"] = df[col].astype(str).apply(
                lambda v: _hash_bucket(v, C.HASH_BUCKETS)
            )

        # 7) Frequency encoding (train-fit maps).
        for col in C.FREQ_COLS:
            df[col + "_freq"] = df[col].map(self.freq_maps_[col]).fillna(0)

        # 8) Boolean-like columns -> {0,1}.
        for col in C.BOOL_COLS:
            df[col] = (
                df[col]
                .replace({"true": 1, "false": 0, "other": 0, True: 1, False: 0})
                .fillna(0)
                .astype(int)
            )

        # 9) scheme_name -> has/hasn't a registered scheme (48% missing).
        df["scheme_name_bool"] = df["scheme_name"].notna().astype(int)

        # 10) One-hot encoding for low cardinality columns.
        df = pd.get_dummies(df, columns=C.ONEHOT_COLS, drop_first=False)

        # 11) Drop redundant / used-up columns.
        drop = list(C.DROP_COLS) + C.HASH_COLS + C.FREQ_COLS + ["scheme_name"]
        df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")

        # Ensure everything is numeric (dummies are bool -> cast to int).
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        return df
