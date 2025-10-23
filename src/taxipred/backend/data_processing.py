import pandas as pd
import numpy as np

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize categorical labels (lowercase, trim)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Convert negative distances/durations to NaN
    for c in ("trip_distance_km", "duration_min"):
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan

    return df
