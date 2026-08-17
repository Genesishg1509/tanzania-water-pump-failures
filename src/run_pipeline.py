"""End-to-end pipeline: load -> features -> train -> calibrate -> evaluate -> decide -> submit.

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
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from . import calibration as CAL
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
    # Three-way split: train (60%) / calib (20%) / test (20%). Calib picks the
    # winning model and calibrates its probabilities; test is touched once, at
    # the very end, so the reported numbers aren't the same set used to choose.
    X_tr_raw, X_rest_raw, y_tr, y_rest = train_test_split(
        X, y, test_size=0.40, random_state=C.RANDOM_STATE, stratify=y
    )
    X_calib_raw, X_test_raw, y_calib, y_test = train_test_split(
        X_rest_raw, y_rest, test_size=0.50, random_state=C.RANDOM_STATE, stratify=y_rest
    )

    # --- 3) Feature engineering (fit on train only) -------------------------
    fe = FeatureEngineer()
    X_tr = fe.fit_transform(X_tr_raw)
    X_calib = fe.transform(X_calib_raw)
    X_test = fe.transform(X_test_raw)
    print(f"Feature matrix: {X_tr.shape[1]} columns after encoding")

    # --- 4) Train on train, evaluate + pick winner on calib ------------------
    metrics, models = {}, {}

    # Baseline (reference only, never "best")
    logreg, scaler = M.train_logistic_regression(X_tr, y_tr)
    models["logreg"] = logreg
    X_calib_scaled = scaler.transform(X_calib)
    metrics["logreg"] = M.evaluate(logreg, X_calib_scaled, y_calib, name="Logistic Regression")

    # Real models (compete for "best")
    if args.model in ("rf", "both"):
        models["rf"] = M.train_random_forest(X_tr, y_tr)
        metrics["rf"] = M.evaluate(models["rf"], X_calib, y_calib, name="Random Forest")
    if args.model in ("lgbm", "both"):
        models["lgbm"] = M.train_lightgbm(X_tr, y_tr, X_calib, y_calib)
        metrics["lgbm"] = M.evaluate(models["lgbm"], X_calib, y_calib, name="LightGBM")
    if args.model in ("both",):  # for now, catboost only with "both"
        models["catboost"] = M.train_catboost(X_tr, y_tr, X_calib, y_calib)
        metrics["catboost"] = M.evaluate(models["catboost"], X_calib, y_calib, name="CatBoost")

    # Only rf, lgbm, catboost compete for best; logreg is reference only
    candidate_keys = [k for k in ["rf", "lgbm", "catboost"] if k in models]
    best_key = max(candidate_keys, key=lambda k: metrics[k]["macro_f1"])
    best = models[best_key]
    print(f"\nBest model by macro F1 (on calib): {metrics[best_key]['name']}")

    print("\nTop features (by importance):")
    print(M.feature_importance(best, X_tr.columns, top=15))

    # --- 5) Calibrate the winner on calib -------------------------------------
    calibrated = CAL.calibrate(best, X_calib, y_calib)

    # --- 6) Everything below is evaluated on TEST, touched once --------------
    proba_raw = best.predict_proba(X_test)
    proba_cal = calibrated.predict_proba(X_test)

    calib_before = CAL.calibration_metrics(y_test, proba_raw)
    calib_after = CAL.calibration_metrics(y_test, proba_cal)

    print("\n===== Calibration (Brier / ECE on test, before vs after) =====")
    for cls in calib_before:
        b0, e0 = calib_before[cls]["brier_score"], calib_before[cls]["ece"]
        b1, e1 = calib_after[cls]["brier_score"], calib_after[cls]["ece"]
        print(f"{cls:>16}: Brier {b0:.4f} -> {b1:.4f}   ECE {e0:.4f} -> {e1:.4f}")

    fig = CAL.reliability_diagram(y_test, proba_cal)
    fig_path = C.FIGURES / "calibration_curve.png"
    CAL.save_reliability_diagram(fig, fig_path)
    print(f"Saved calibration figure -> {fig_path}")

    print("\n===== Final evaluation on TEST (calibrated model) =====")
    final_metrics = M.evaluate(calibrated, X_test, y_test,
                               name=f"{metrics[best_key]['name']} (calibrated)")

    # --- 7) Decision layer: what the model is actually worth (on test) -------
    print("\n===== Risk ranking (limited inspection budget) =====")
    ranking = DEC.ranking_report(y_test, proba_cal)
    print(ranking.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n===== Decision rule: accuracy-optimal vs cost-optimal =====")
    costs = DEC.cost_comparison(y_test, proba_cal)
    print(costs.to_string(float_format=lambda v: f"{v:,.1f}"))

    # --- 8) Persist artifacts --------------------------------------------------
    if args.save_model:
        import joblib
        C.MODELS.mkdir(parents=True, exist_ok=True)
        out = C.MODELS / f"{best_key}_pipeline.joblib"
        joblib.dump({"model": calibrated, "feature_engineer": fe,
                     "columns": list(X_tr.columns)}, out)
        print(f"\nSaved model -> {out}")

    if args.metrics_json:
        C.REPORTS.mkdir(parents=True, exist_ok=True)
        out = C.REPORTS / "metrics.json"
        out.write_text(json.dumps(
            {"models": metrics, "best": best_key,
             "final_test_metrics": final_metrics,
             "calibration": {"before": calib_before, "after": calib_after},
             "ranking": ranking.reset_index().to_dict(orient="records"),
             "cost": costs.reset_index().to_dict(orient="records")},
            indent=2), encoding="utf-8")
        print(f"Saved metrics -> {out}")

    # --- 9) Competition submission ----------------------------------------------
    if not args.no_submission:
        comp = D.load_competition()
        X_comp = fe.transform(comp)
        submission = pd.DataFrame({
            "id": comp["id"],
            "status_group": D.decode_target(calibrated.predict(X_comp)),
        })
        out = C.ROOT / "submission.csv"
        submission.to_csv(out, index=False)
        print(f"Saved submission -> {out}")

    return metrics


if __name__ == "__main__":
    main()
