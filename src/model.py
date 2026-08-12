"""Model training and evaluation.

Two tree-based classifiers are provided. Both use ``class_weight`` to handle
the class imbalance instead of the original notebook's SMOTE step.

Why drop SMOTE?  In the original code SMOTE ran *after* one-hot encoding, so it
synthesised fractional values (e.g. 0.37) inside binary dummy columns and
interpolated the nominal hash-bucket ids — neither is meaningful. It was also
combined with ``class_weight`` in LightGBM, correcting the imbalance twice.
For tree ensembles, per-class weighting is cleaner and usually at least as
strong, so we keep a single, well-understood mechanism.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             roc_auc_score)

from . import config as C


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=C.RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, X_val=None, y_val=None) -> LGBMClassifier:
    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=70,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=C.RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    fit_kwargs = {}
    if X_val is not None:
        from lightgbm import early_stopping, log_evaluation
        fit_kwargs = dict(
            eval_X=X_val,
            eval_y=y_val,
            eval_metric="multi_logloss",
            callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
        )
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def evaluate(model, X_val, y_val, name: str = "model") -> dict:
    """Print a report and return key metrics as a dict."""
    y_pred = model.predict(X_val)
    labels = [0, 1, 2]

    acc = (y_pred == y_val).mean()
    macro_f1 = f1_score(y_val, y_pred, average="macro", labels=labels)
    f1_repair = f1_score(y_val, y_pred, labels=[2], average="macro")

    try:
        proba = model.predict_proba(X_val)
        auc = roc_auc_score(y_val, proba, multi_class="ovr", labels=labels)
    except Exception:
        auc = float("nan")

    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print(f"F1 (needs repair) : {f1_repair:.4f}")
    print(f"ROC AUC (OVR)     : {auc:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred) labels=[0,1,2]:")
    print(confusion_matrix(y_val, y_pred, labels=labels))
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, labels=labels, digits=4,
                                zero_division=0))

    return {"name": name, "accuracy": acc, "macro_f1": macro_f1,
            "f1_needs_repair": f1_repair, "roc_auc": auc}


def feature_importance(model, feature_names, top: int = 20) -> pd.Series:
    """Return the top-N features by importance (gain for LGBM)."""
    if hasattr(model, "booster_"):
        imp = model.booster_.feature_importance(importance_type="gain")
    else:
        imp = model.feature_importances_
    return (pd.Series(imp, index=feature_names)
            .sort_values(ascending=False)
            .head(top))
