"""Data loading and target encoding.

Pure local file I/O — no Google Colab / Drive dependencies, so the project
runs on any machine after `download_data` (or a manual copy) puts the CSVs in
`data/raw/`.
"""
from __future__ import annotations

import pandas as pd

from . import config as C


def load_training() -> pd.DataFrame:
    """Load the merged training set (features + label)."""
    if not C.TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Missing {C.TRAIN_FILE}. See the README 'Data' section on how to "
            "obtain training_merged.csv from DrivenData."
        )
    return pd.read_csv(C.TRAIN_FILE)


def load_competition() -> pd.DataFrame:
    """Load the unlabeled competition set used for the DrivenData submission."""
    if not C.COMP_FILE.exists():
        raise FileNotFoundError(
            f"Missing {C.COMP_FILE}. See the README 'Data' section."
        )
    return pd.read_csv(C.COMP_FILE)


def encode_target(y: pd.Series) -> pd.Series:
    """Map the three text labels to integer codes (see config.LABEL_TO_CODE)."""
    return y.map(C.LABEL_TO_CODE).astype(int)


def decode_target(y_codes) -> pd.Series:
    """Map integer codes back to the original DrivenData text labels."""
    return pd.Series(y_codes).map(C.CODE_TO_LABEL)
