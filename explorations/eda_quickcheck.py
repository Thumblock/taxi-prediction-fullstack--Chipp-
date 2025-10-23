# explorations/eda_quickcheck.py
# Run: uv run python explorations/eda_quickcheck.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from taxipred.utils.constants import TAXI_CSV_PATH, TARGET_COL

OUT = Path("explorations/figs")
OUT.mkdir(parents=True, exist_ok=True)

print("CSV path:", TAXI_CSV_PATH)
print("Target  :", TARGET_COL)

df = pd.read_csv(TAXI_CSV_PATH)
print("Shape:", df.shape)
print(df.head(3))

print("\nDtypes:")
print(df.dtypes)

print("\nMissing fraction by column:")
print(df.isna().mean().sort_values(ascending=False).head(20))

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
print("\nNumeric cols:", numeric_cols)
print("Categorical cols:", cat_cols)

# Save a target histogram if present
if TARGET_COL in df.columns:
    ax = df[TARGET_COL].dropna().plot(kind="hist", bins=40, title=f"Histogram: {TARGET_COL}")
    ax.figure.savefig(OUT / f"hist_{TARGET_COL}.png", bbox_inches="tight")
    plt.close(ax.figure)

# A few numeric hist/box plots
for col in numeric_cols[:8]:
    ax = df[col].dropna().plot(kind="hist", bins=40, title=f"Histogram: {col}")
    ax.figure.savefig(OUT / f"hist_{col}.png", bbox_inches="tight")
    plt.close(ax.figure)

for col in numeric_cols[:8]:
    ax = df[col].dropna().plot(kind="box", title=f"Box: {col}")
    ax.figure.savefig(OUT / f"box_{col}.png", bbox_inches="tight")
    plt.close(ax.figure)

# Cardinality and top values for categoricals
card = {c: int(df[c].nunique(dropna=True)) for c in cat_cols}
card_df = pd.DataFrame.from_dict(card, orient="index", columns=["n_unique"]).sort_values("n_unique", ascending=False)
print("\nCategorical cardinality (top 10):")
print(card_df.head(10))

for c in cat_cols[:10]:
    print(f"\n== {c} ==")
    print(df[c].value_counts(dropna=False).head(10))

# With/without label split
if TARGET_COL in df.columns:
    no_label = df[df[TARGET_COL].isna()]
    with_label = df[df[TARGET_COL].notna()]
    print("\nRows WITHOUT label:", no_label.shape, "— keep for app inputs later")
    print("Rows WITH label   :", with_label.shape, "— used for training")
