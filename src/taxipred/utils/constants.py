from importlib.resources import files
from pathlib import Path


TAXI_CSV_PATH = files("taxipred").joinpath("data/taxi_trip_pricing.csv")  
TARGET_COL    = "price"                      # label name

MODEL_DIR = Path(TAXI_CSV_PATH).parent
MODEL_PATH = MODEL_DIR / "model.joblib"      # pipeline (preprocess + model)
META_PATH  = MODEL_DIR / "model_meta.json"   # metrics + best model info