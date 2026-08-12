"""End-to-end pipeline: load -> features -> train -> evaluate -> decide -> submit.

Run with::

    python -m src.run_pipeline                  # train both models, full report
    python -m src.run_pipeline --model lgbm     # a single model
    python -m src.run_pipeline --save-model     # persist the winner to models/

Reproducible and free of any Colab / Google Drive dependency.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C
from . import data as D
from . import decision as DEC
from . import model as M
from .features import FeatureEngineer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.run_pipeline",
        description="Train and evaluate the Tanzania water-pump status models.",
    )
    p.add_argument("--model", choices=["rf", "lgbm", "both"], default="both",
                   help="which model(s) to train (default: both)")
    p.add_argument("--save-model", action="store_true",
                   help="persist the winning model + fitted encoders to models/")
    p.add_argument("--no-submission", action="store_true",
                   help="skip writing the DrivenData submission file")
    p.add_argument("--metrics-json", action="store_true",
                   help="write the evaluation metrics to reports/metrics.json")
    return p


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)

    # --- 1) Load ------------------------------------------------------------
    df = D.load_training()
    y = D.encode_target(df[C.TARGET])
    X = df.drop(columns=[C.TARGET])
    print(f"Training rows: {len(df):,}")

    # --- 2) Split BEFORE fitting any encoder (prevents leakage) -------------
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE, stratify=y
    )

    # --- 3) Feature engineering (fit on train only) -------------------------
    fe = FeatureEngineer()
    X_tr = fe.fit_transform(X_tr_raw)
    X_val = fe.transform(X_val_raw)
    print(f"Feature matrix: {X_tr.shape[1]} columns after encoding")

    # --- 4) Train + evaluate ------------------------------------------------
    metrics, models = {}, {}
    if args.model in ("rf", "both"):
        models["rf"] = M.train_random_forest(X_tr, y_tr)
        metrics["rf"] = M.evaluate(models["rf"], X_val, y_val, name="Random Forest")
    if args.model in ("lgbm", "both"):
        models["lgbm"] = M.train_lightgbm(X_tr, y_tr, X_val, y_val)
        metrics["lgbm"] = M.evaluate(models["lgbm"], X_val, y_val, name="LightGBM")

    best_key = max(metrics, key=lambda k: metrics[k]["macro_f1"])
    best = models[best_key]
    print(f"\nBest model by macro F1: {metrics[best_key]['name']}")

    print("\nTop features (by importance):")
    print(M.feature_importance(best, X_tr.columns, top=15))

    # --- 5) Decision layer: what the model is actually worth ----------------
    proba = best.predict_proba(X_val)

    print("\n===== Risk ranking (limited inspection budget) =====")
    ranking = DEC.ranking_report(y_val, proba)
    print(ranking.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n===== Decision rule: accuracy-optimal vs cost-optimal =====")
    costs = DEC.cost_comparison(y_val, proba)
    print(costs.to_string(float_format=lambda v: f"{v:,.1f}"))

    # --- 6) Persist artifacts ----------------------------------------------
    if args.save_model:
        import joblib
        C.MODELS.mkdir(parents=True, exist_ok=True)
        out = C.MODELS / f"{best_key}_pipeline.joblib"
        joblib.dump({"model": best, "feature_engineer": fe,
                     "columns": list(X_tr.columns)}, out)
        print(f"\nSaved model -> {out}")

    if args.metrics_json:
        C.REPORTS.mkdir(parents=True, exist_ok=True)
        out = C.REPORTS / "metrics.json"
        out.write_text(json.dumps(
            {"models": metrics, "best": best_key,
             "ranking": ranking.reset_index().to_dict(orient="records"),
             "cost": costs.reset_index().to_dict(orient="records")},
            indent=2), encoding="utf-8")
        print(f"Saved metrics -> {out}")

    # --- 7) Competition submission -----------------------------------------
    if not args.no_submission:
        comp = D.load_competition()
        X_comp = fe.transform(comp)
        submission = pd.DataFrame({
            "id": comp["id"],
            "status_group": D.decode_target(best.predict(X_comp)),
        })
        out = C.ROOT / "submission.csv"
        submission.to_csv(out, index=False)
        print(f"Saved submission -> {out}")

    return metrics


if __name__ == "__main__":
    main()
