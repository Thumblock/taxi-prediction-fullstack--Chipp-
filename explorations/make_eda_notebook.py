import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

# Ensure folder exists
os.makedirs("explorations", exist_ok=True)

# Build a list of cells
cells = []

# Cell 1: Title (Markdown)
cells.append(new_markdown_cell("# TaxiPred — EDA"))

# Cell 2: Imports & constants
cell_imports = [
    "import pandas as pd",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "",
    "from taxipred.utils.constants import TAXI_CSV_PATH, TARGET_COL",
    "",
    "pd.set_option('display.max_columns', 100)",
    "print('CSV path:', TAXI_CSV_PATH)",
    "print('Target column:', TARGET_COL)",
]
cells.append(new_code_cell("\n".join(cell_imports)))

# Cell 3: Load & preview 
cell_load = [
    "df = pd.read_csv(TAXI_CSV_PATH)",
    "print('Shape:', df.shape)",
    "df.head()",
]
cells.append(new_code_cell("\n".join(cell_load)))

# Cell 4: Schema & numeric summary
cell_schema = [
    "print('dtypes:')",
    "print(df.dtypes)",
    "print('\\n.info():')",
    "print(df.info())",
    "df.describe(include=[np.number]).T",
]
cells.append(new_code_cell("\n".join(cell_schema)))

# Cell 5: Missingness 
cell_nulls = [
    "nulls = df.isna().mean().sort_values(ascending=False)",
    "nulls.to_frame('null_fraction')",
]
cells.append(new_code_cell("\n".join(cell_nulls)))

# Cell 6: Quick target histogram
cell_hist = [
    "if TARGET_COL in df.columns:",
    "    ax = df[TARGET_COL].dropna().plot(kind='hist', bins=40, "
    "title=f'Histogram: {TARGET_COL}')",
    "    plt.xlabel(TARGET_COL)",
    "    plt.ylabel('count')",
    "    plt.show()",
    "else:",
    "    print(f\"Target column '{TARGET_COL}' not found.\")",
]
cells.append(new_code_cell("\n".join(cell_hist)))

# Assemble and save the notebook
nb = new_notebook(
    cells=cells,
    metadata={"kernelspec": {"display_name": "Python 3", "name": "python3"}},
)

with open("explorations/eda.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Wrote explorations/eda.ipynb (clean)")
