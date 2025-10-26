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