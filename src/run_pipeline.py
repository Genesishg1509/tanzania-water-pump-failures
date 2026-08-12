"""End-to-end pipeline: load -> features -> train -> evaluate -> submission.

Run with:  python -m src.run_pipeline
Reproducible and free of any Colab / Google Drive dependency.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C
from . import data as D
from . import model as M
from .features import FeatureEngineer


def main() -> None:
    # --- 1) Load ------------------------------------------------------------
    df = D.load_training()
    comp = D.load_competition()
    print(f"Training rows: {len(df):,} | Competition rows: {len(comp):,}")

    y = D.encode_target(df[C.TARGET])
    X = df.drop(columns=[C.TARGET])

    # --- 2) Split BEFORE fitting any encoder (prevents leakage) -------------
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE, stratify=y
    )

    # --- 3) Feature engineering (fit on train only) -------------------------
    fe = FeatureEngineer()
    X_tr = fe.fit_transform(X_tr_raw)
    X_val = fe.transform(X_val_raw)
    X_comp = fe.transform(comp)
    print(f"Feature matrix: {X_tr.shape[1]} columns after encoding")

    # --- 4) Train + evaluate ------------------------------------------------
    rf = M.train_random_forest(X_tr, y_tr)
    rf_metrics = M.evaluate(rf, X_val, y_val, name="Random Forest")

    lgbm = M.train_lightgbm(X_tr, y_tr, X_val, y_val)
    lgbm_metrics = M.evaluate(lgbm, X_val, y_val, name="LightGBM")

    print("\nTop features (LightGBM, by gain):")
    print(M.feature_importance(lgbm, X_tr.columns, top=15))

    # --- 5) Competition submission (best model) -----------------------------
    best = lgbm if lgbm_metrics["macro_f1"] >= rf_metrics["macro_f1"] else rf
    C.MODELS.mkdir(parents=True, exist_ok=True)
    preds = D.decode_target(best.predict(X_comp))
    submission = pd.DataFrame({"id": comp["id"], "status_group": preds})
    out = C.ROOT / "submission.csv"
    submission.to_csv(out, index=False)
    print(f"\nSaved submission -> {out}")

    return {"rf": rf_metrics, "lgbm": lgbm_metrics}


if __name__ == "__main__":
    main()
