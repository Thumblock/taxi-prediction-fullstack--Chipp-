# Loads the trained sklearn Pipeline once (preprocessor + model)
#   - Endpoints :
#  - GET  /Health  -> status + if model is loaded.
#  - POST /predict -> predict price from JSON

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib   

from taxipred.utils.constants import MODEL_PATH

app = FastAPI(title="TaxiPred API", version="0.1.0")

# Load once
# sklearn Pipeline(preprocessor+model)
try:
    MODEL_PIPE = joblib.load(MODEL_PATH)  
    print(f"[startup] Loaded model from {MODEL_PATH}")
except Exception as e:
    MODEL_PIPE = None
    print(f"[startup] Failed to load model: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded" : MODEL_PIPE is not None}

class Predict_Trip(BaseModel):
    Trip_Distance_km: float
    Passenger_Count: int
    Time_of_Day: str
    Day_of_Week: str
    Traffic_Conditions: str
    Weather: str

@app.post("/predict")
def predict(payload: Predict_Trip):
    if MODEL_PIPE is None:
        raise HTTPException(status_code=503, detail="Model not loaded (run training first)")
    df = pd.DataFrame([payload.model_dump()])   
    yhat = MODEL_PIPE.predict(df)
    return {"predicted_price": float(yhat[0])}