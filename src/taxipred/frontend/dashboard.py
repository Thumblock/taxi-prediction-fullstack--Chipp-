import streamlit as st
import requests

st.set_page_config(page_title="Taxi Price Predictor", page_icon="🚕")
st.title("🚕 Taxi Price Predictor")
st.markdown("Enter Trip Detail and get an estimated price.")

API_URL = "http://127.0.0.1:8000/predict"

distance = st.number_input("Trip Distance (km)", min_value=0.0, step=0.1)
passengers = st.number_input("Passenger Count", min_value=1, step=1)

time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "evening", "night"])
day_of_week = st.selectbox("Day of Week", ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])
traffic = st.selectbox("Traffic Conditions", ["light", "moderate", "heavy"])
weather = st.selectbox("weather", ["clear", "rainy", "snowy", "foggy"])

if st.button("Predict Price 💰"):
    payload = {
        "Trip_Distance_km": distance,
        "Passenger_Count": passengers,
        "Time_of_Day": time_of_day,
        "Day_of_Week": day_of_week,
        "Traffic_Conditions": traffic,
        "Weather": weather
    }

    try:
        res = requests.post(API_URL, json=payload, timeout=10)
        if res.ok:
            price = res.json().get("predicted_price")
            st.success(f"Estimated Price: **${price:.2f}**")
        else:
            st.error(f"Error {res.status_code}: {res.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")