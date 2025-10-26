from importlib.resources import files
from pathlib import Path


TAXI_CSV_PATH = files("taxipred").joinpath("data/taxi_trip_pricing.csv")  
TARGET_COL    = "Trip_Price"                      # label name

#Training set
FEATURE_COLUMNS = [
    "Trip_Distance_km",
    "Passenger_Count",
    "Time_of_Day",
    "Day_of_Week",
    "Traffic_Conditions",
    "Weather"
]

MODEL_DIR = Path(TAXI_CSV_PATH).parent
MODEL_PATH = MODEL_DIR / "model.joblib"      # pipeline (preprocess + model)
META_PATH  = MODEL_DIR / "model_meta.json"   # metrics + best model info