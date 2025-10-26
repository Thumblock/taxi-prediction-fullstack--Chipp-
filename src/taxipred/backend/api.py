from fastapi import FastAPI, HTTPException
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