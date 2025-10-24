# Goal of this module (step 1):
# - Load CSV (via TAXI_CSV_PATH), apply the same cleaning functions. 
# - Keep only rows that have labels for supervised learning
# - Split into X (features) and y (target), ready for ML
# - NOTE: In AI, inference is the process where a trained machine learning model uses its knowledge to make predictions or decisions on new, unseen data.


import pandas as pd

# Project-wide single sources of truth:
from taxipred.utils.constants import TAXI_CSV_PATH, TARGET_COL

# Reuse shared, consistent data prep (train == inference):
# reuse the same cleaning/transformation rules everywhere (training and later in the API), avoiding train/serve skew.
from taxipred.backend.data_processing import (
    clean_and_engineer,
    split_labeled_unlabeled,
    prepare_xy,
)

# Load dataset from packaged path
df = pd.read_csv(TAXI_CSV_PATH)
print("Loaded:", TAXI_CSV_PATH, "| shape:", df.shape)

# Apply package-wide cleaning (normalize strings, fix negatives, clip outliers)
df = clean_and_engineer(df)

# Supervised training: keep only rows that have a label/target
with_label, no_label = split_labeled_unlabeled(df, TARGET_COL)
print("with_label:", with_label.shape, "| no_label (for future predictions):", no_label.shape)

# Split features/target — X for inputs, y for ground-truth label
# If TARGET_COL is missing, prepare_xy raises a clear KeyError (better to fail early)
X, y = prepare_xy(with_label, TARGET_COL)
print("Prepared X/y:", X.shape, y.shape)