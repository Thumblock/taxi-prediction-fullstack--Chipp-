# 🚕 TaxiPred — Taxi Price Prediction (Fullstack Project)

Goal: Build an end-to-end ML app to predict taxi prices — EDA → model → API → UI.  

> **Author:** Chipp Larusson  
> **Course:** lab_taxipred — Full-stack ML Application  
> **Purpose:** Create a complete ML system that predicts taxi trip prices using real-world-like features and serves it through a FastAPI backend and Streamlit frontend.

---

## 📘 Overview

This project predicts taxi prices based on trip details such as:

- Distance (km)  
- Passenger count  
- Time of day  
- Day of week  
- Traffic conditions  
- Weather  


It combines:
- 🧹 **Data cleaning & preprocessing** (`data_processing.py`)
- 🤖 **ML model training & evaluation** (`model_training.py`)
- 🌐 **Backend API** for live predictions (`api.py`)
- 🖥️ **Streamlit dashboard** for user interaction (`dashboard.py`)


Both backend and frontend share the same **6-feature schema**, ensuring consistency between training and serving.

---

## 🧩 Project Steps

1. Load & clean data (normalize, clip, handle negatives).  
2. Train two models (Linear Regression, RandomForest).  
3. Evaluate by RMSE and select the best model.  
4. Save preprocessing + model pipeline (`model.joblib`).  
5. Serve predictions via FastAPI.  
6. Interact with the model through the Streamlit dashboard.

---

## 🔧 How to Run (using `uv`)

### 📦 1. Create & activate virtual environment
````bash
uv venv

# Activate:
# Windows : Bash
source .venv/Scripts/activate
# macOS/Linux
source .venv/bin/activate

### 📦 2. Install the project in editable mode
```bash
uv pip install -e .

📊 3. Run quick EDA scripts
uv run python explorations/eda_quickcheck.py
# (optional)
uv run python explorations/make_eda_notebook.py

🧠 4. Train the ML model
uv run python src/taxipred/backend/model_training.py

⚙️ 5. Start FastAPI backend
uv run uvicorn taxipred.backend.api:app --reload
Visit: http://127.0.0.1:8000/health
Expected response: {"status": "ok", "model_loaded": true}