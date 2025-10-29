# 🚕 TaxiPred — Taxi Price Prediction (Fullstack Project)

Goal: build E2E ML app to predict taxi prices — EDA → model → API → UI. 

> **Author:** Chipp Larusson
> **Course:** lab_taxipred — Full-stack ML Application  
> **Purpose:** Build an end-to-end system that predicts taxi trip prices using real-world-like features and serves it via a FastAPI backend + Streamlit frontend.

## 📘 Overview

This project predicts taxi prices based on trip details such as distance, passenger count, time of day, weekday, traffic, and weather conditions.  

It combines:
- 🧹 **Data cleaning & preprocessing** (`data_processing.py`)
- 🤖 **ML model training & evaluation** (`model_training.py`)
- 🌐 **Backend API** for live predictions (`api.py`)
- 🖥️ **Streamlit dashboard** for user interaction (`dashboard.py`)

The backend and frontend share the same 6-feature schema for consistent predictions.


---

## 📂 Project Step

1. Load & clean data (normalize, clip, handle negatives)

2. Train 2 models → evaluate → select best by RMSE

3. Save entire preprocessing + model pipeline

4. Serve predictions with FastAPI

5. Interact through Streamlit frontend

---


## 🔧 How to Run

1. create venv (if you don't already have one)
uv venv
# Git Bash
source .venv/Scripts/activate
# or macOS/Linux
source .venv/bin/activate

**Install packages**
# install your package in editable mode
uv pip install -e .

## EDA 
uv run python explorations/eda_quickcheck.py
# (optional)Generate a clean EDA notebook.py
uv run python explorations/make_eda_notebook.py

# Train the model (Linear Regression vs RandomForest)
uv run python src/taxipred/backend/model_training.py

# Start the API
uv run uvicorn taxipred.backend.api:app --reload
** Sanity check the API: **
Health:
Browser: http://127.0.0.1:8000/health
Expected: {"status":"ok","model_loaded":true}

# Run the Streamlit dashboard
uv run streamlit run src/taxipred/frontend/dashboard.py


----

# Full Project Sanity Checklist

Training runs end-to-end and saves artifacts:

src/taxipred/data/model.joblib created ✅

src/taxipred/data/model_meta.json created ✅

“Sanity check passed” printed ✅

API boots and loads model:

/health → model_loaded: true ✅

Prediction works:

Python or curl returns {"predicted_price": ...} ✅

Dashboard works:

UI submits payload → shows price ✅