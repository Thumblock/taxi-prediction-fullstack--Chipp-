#Just a basic FastAPI to check it works with helpers.py in terminal.

from fastapi import FastAPI

app = FastAPI(title="TaxiPred (stub)", version="0.0.1")

@app.get("/health")
def health():
    return {"status": "ok"}