import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Local package:
from taxipred.utils.constants import TARGET_COL  # single source of truth

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize categorical labels (lowercase, trim)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Convert negative distances/durations to NaN
    for c in ("trip_distance_km", "duration_min"):
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan

    # Clip extreme numeric values to 1%–99% range
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        s = df[c].dropna()
        if len(s) >= 10:  # enough data to compute quantiles
            lo, hi = s.quantile([0.01, 0.99])
            df[c] = df[c].clip(lower=lo, upper=hi)

    return df

#Return (with_label, no_label) DataFrames. Rows with NaN target will be used for predictions later.
def split_labeled_unlabeled(df: pd.DataFrame, target_col: str = TARGET_COL):
    
    if target_col not in df.columns:
        return df.copy(), df.iloc[0:0].copy()

    with_label = df[df[target_col].notna()].copy()
    no_label = df[df[target_col].isna()].copy()
    return with_label, no_label

#Infer numeric and categorical feature columns by dtype.
def infer_feature_columns(df: pd.DataFrame, target_col: str = TARGET_COL):

    # Numeric feature columns (float/int)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    # Categorical feature columns (object/category)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    return num_cols, cat_cols


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    # Numeric pipeline: fill NaNs with median, then scale to zero mean / unit variance
    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    # Categorical pipeline: fill NaNs with mode, then one-hot encode
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # ColumnTransformer applies the right pipeline to each column subset
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, list(numeric_cols)),
            ("cat", cat_pipe, list(categorical_cols)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre

# Split features X and target y for model training
def prepare_xy(df: pd.DataFrame, target_col: str = TARGET_COL):
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} not found in DataFrame.")
    y = pd.to_numeric(df[target_col], errors="coerce")
    X = df.drop(columns=[target_col])
    return X, y
