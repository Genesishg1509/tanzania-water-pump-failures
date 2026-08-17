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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

from . import config as C


def train_logistic_regression(X_train, y_train) -> tuple[LogisticRegression, StandardScaler]:
    """Linear baseline (requires scaling). Shows importance of tree nonlinearity."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=C.RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
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


def train_catboost(X_train, y_train, X_val=None, y_val=None) -> CatBoostClassifier:
    """Third boosting algorithm. Receives already-encoded features (no categorical advantage here)."""
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=8,
        class_weights={0: 1, 1: 1, 2: 3},  # weight rare class
        random_state=C.RANDOM_STATE,
        verbose=False,
    )
    eval_set = None
    if X_val is not None:
        eval_set = [(X_val, y_val)]
    
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_val, y_val, name: str = "model") -> dict:
    """Print a report and return key metrics as a dict."""
    y_pred = np.asarray(model.predict(X_val)).reshape(-1)
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
