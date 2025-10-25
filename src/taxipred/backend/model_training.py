# Goal of this module (step 3):
# - After building preprocessor, train two models (LR, RF) with identical preprocessing.
# - Print metrics: MAE, RMSE, R².


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
from taxipred.utils.constants import TAXI_CSV_PATH, TARGET_COL

# Reuse shared, consistent data prep (train == inference):
# reuse the same cleaning/transformation rules everywhere (training and later in the API), avoiding train/serve skew.
from taxipred.backend.data_processing import (
    clean_and_engineer,
    split_labeled_unlabeled,
    prepare_xy,
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

    # Split features/target — X for inputs, y for ground-truth label
    X, y = prepare_xy(with_label, TARGET_COL)
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

if __name__ == "__main__":
    main()