# Tanzania Water Pump Failures — Predicting & Prioritizing Repairs

Thousands of communities in Tanzania rely on water pumps for daily access to
clean water. A large share of these pumps are broken or at risk of failing.
This project builds a machine-learning model that predicts the operational
status of each pump so that authorities can **prioritize maintenance and target
inspection resources where they matter most**.

Data comes from the [DrivenData "Pump it Up"](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/)
competition: ~59,400 labeled pumps, 40 features, and a 3-class target
(`functional`, `non functional`, `functional needs repair`).

## Results

Evaluated on a stratified 20% hold-out split. `needs repair` is the rare,
high-value minority class (7% of pumps) and the hardest to predict.

| Model | Accuracy | Macro F1 | F1 (needs repair) | ROC AUC (OVR) |
|-------|:--------:|:--------:|:-----------------:|:-------------:|
| Random Forest | 0.798 | **0.704** | 0.457 | **0.904** |
| LightGBM      | 0.787 | 0.699 | **0.460** | 0.902 |

**Most predictive signals:** water `quantity` (a *dry* pump is very likely
non-functional), geographic location (`latitude`/`longitude`, local
`ward`/`lga`), pump age (`years_in_operation`, `construction_year`) and who
funded/installed it.

> These numbers come from a **leak-free** pipeline. An earlier version leaked
> competition data into the encoders; removing that leak — and dropping an
> unjustified `log1p` transform and a mis-applied SMOTE step — the honest model
> actually scores *higher* on the difficult minority class than the leaky one.

## Project structure

```
├── data/
│   ├── raw/                # training_merged.csv, test_set_values_x_test.csv (not versioned)
│   └── processed/
├── notebooks/              # narrative EDA + storytelling
├── src/
│   ├── config.py           # paths, column groups, constants
│   ├── data.py             # loading + target encoding (no Colab dependency)
│   ├── features.py         # leak-free feature engineering (fit on train only)
│   ├── model.py            # Random Forest & LightGBM + evaluation
│   └── run_pipeline.py     # end-to-end: load → features → train → submit
├── reports/figures/
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. Install dependencies (a virtual environment is recommended)
pip install -r requirements.txt

# 2. Place the two CSVs in data/raw/
#    - training_merged.csv           (X_train merged with y_train on `id`)
#    - test_set_values_x_test.csv    (competition set, no labels)
#    Both are available from the DrivenData competition page linked above.

# 3. Run the full pipeline
python -m src.run_pipeline
```

This trains both models, prints an evaluation report, and writes
`submission.csv` for the competition.

## Method

1. **Cleaning** — text normalization; zeros that actually mean "missing"
   (`amount_tsh`, `construction_year`, `district_code`, `longitude`) imputed
   from **train-only** statistics (median / mode / conditional median by region).
2. **Feature engineering** — `years_in_operation` from record vs. construction
   year; `scheme_name` → has-a-scheme flag.
3. **Encoding by cardinality** —
   - very high (`subvillage`) → feature hashing,
   - high (`ward`, `lga`, `funder`, `installer`) → frequency encoding (train-fit),
   - low (`region`, `basin`, `quantity`, …) → one-hot,
   - redundant hierarchy duplicates dropped.
4. **Imbalance** — handled with per-class weighting (`class_weight`) rather than
   SMOTE, which is cleaner for tree ensembles and avoids synthesizing
   fractional values inside binary/one-hot columns.
5. **Models** — Random Forest and LightGBM (early stopping on the hold-out).

## Roadmap

- [ ] Hyper-parameter search with Optuna + stratified cross-validation
- [ ] Cost-sensitive decision layer (a missed `needs repair` is costlier than a
      false alarm) and a risk-ranking view (precision@k / lift)
- [ ] Interactive map of predicted pump risk across Tanzania
- [ ] Streamlit app for exploring at-risk pumps by region
