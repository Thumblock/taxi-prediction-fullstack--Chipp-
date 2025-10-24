# Goal of this module (step 2):
# - Load CSV (via TAXI_CSV_PATH), apply the same cleaning functions. 
# - Keep only rows that have labels for supervised learning
# - Split into X (features) and y (target), ready for ML
# - Add: train/test split, infer numeric/categorical features, build shared preprocessor.


import pandas as pd
from sklearn.model_selection import train_test_split 

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

if __name__ == "__main__":
    main()