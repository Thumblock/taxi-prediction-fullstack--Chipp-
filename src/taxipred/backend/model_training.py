# Goal of this module (step 4):
# - Pick best model (lowest RMSE), refit on all labeled data.
# - Save trained pipeline + metadata.
# - Sanity check — reload saved pipeline and run one prediction.

import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# RMSE import helper
try:
    from sklearn.metrics import root_mean_squared_error as _rmse
    def rmse(y_true, y_pred):
        return float(_rmse(y_true, y_pred))
except Exception:
    from sklearn.metrics import mean_squared_error as _mse
    def rmse(y_true, y_pred):
        return float(np.sqrt(_mse(y_true, y_pred)))

# Project-wide single sources of truth:
from taxipred.utils.constants import TAXI_CSV_PATH, TARGET_COL, MODEL_PATH, META_PATH, FEATURE_COLUMNS

# Reuse shared, consistent data prep (train == inference):
# reuse the same cleaning/transformation rules everywhere (training and later in the API), avoiding train/serve skew.
from taxipred.backend.data_processing import (
    clean_and_engineer,
    split_labeled_unlabeled,
    infer_feature_columns,
    build_preprocessor,
)

def main() -> None:
    # Load dataset from packaged path
    df = pd.read_csv(TAXI_CSV_PATH)
    print("Loaded:", TAXI_CSV_PATH, "| shape:", df.shape)

    # Apply package-wide cleaning (normalize strings, fix negatives, clip outliers)
    df = clean_and_engineer(df)

    # Supervised training: keep only rows that have a label/target
    with_label, no_label = split_labeled_unlabeled(df, TARGET_COL)
    print("with_label:", with_label.shape, "| no_label (for future predictions):", no_label.shape)
    if with_label.empty:
        raise RuntimeError("No Labeled rows found after cleaning.")

    # Split features/target - X for inputs, y for ground-truth label
    X = with_label[FEATURE_COLUMNS].copy()
    y = pd.to_numeric(with_label[TARGET_COL], errors="coerce")
    print("Prepared X/y:", X.shape, y.shape)

    # Train/test split for honest evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Infer numeric/categorical feature columns
    num_cols, cat_cols = infer_feature_columns(X_train, TARGET_COL)
    print("Numeric columns:", num_cols)
    print("Categorical columns:", cat_cols)
          
    # Build preprocessor (Shared : training and inference)
    pre = build_preprocessor(num_cols,cat_cols)
    print("Preprocessor ready.")

    # Define candidate models
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        ),
    }

    # Small helper to compute metrics
    def evaluate(pipe, Xte, yte):
        pred = pipe.predict(Xte)
        return {
            "MAE": mean_absolute_error(yte, pred),
            "RMSE": rmse(yte, pred),
            "R2": r2_score(yte, pred),
        }
    results = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", pre),("model", model)])
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        results[name] = metrics
        print(f"== {name} ==")
        print(f"MAE : {metrics['MAE']:.3f}")
        print(f"RMSE : {metrics['RMSE']:.3f}")
        print(f"R2 : {metrics['R2']:.4f}")

    # Pick best model
    best_name = min(results, key=lambda k: results[k]["RMSE"])
    best_metrics = results[best_name]
    print(f" Best model: {best_name} (RMSE={best_metrics['RMSE']:.3f})")

    # Refit on all labeled data
    best_model = models[best_name]
    pipe_final = Pipeline([("prep", pre),("model", best_model)])
    pipe_final.fit(X, y)
    print("Refitted best model on full labeled dataset")

    # Save pipeline and metadata
    joblib.dump(pipe_final, MODEL_PATH)
    print(f"Pipeline fully saved to {MODEL_PATH}")

    meta = {
        "best_model": best_name,    
        "metrics": best_metrics,
        "model_path": str(MODEL_PATH)
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Checked : metadata to [META_PATH]")

    # Sanity check: Reload and predict one sample
    print("Sanity check done :")
    pipe_loaded = joblib.load(MODEL_PATH)

    sample = X.sample(1, random_state=42)
    pred = pipe_loaded.predict(sample)
    true_val = y.loc[sample.index].iloc[0]

    print(f"Sample true value : {true_val:.2f}")
    print(f"Predicted value : {pred[0]:.2f}")
    print("Sanity checkk passed - pipeline loads and predicts succesfully")

if __name__ == "__main__":
    main()