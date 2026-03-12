import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
from pathlib import Path

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="OLA Demand Forecasting",
    page_icon="🚕",
    layout="wide"
)

st.title("🚕 OLA Ride Demand Forecasting")
st.write("Predict expected ride demand using the trained ML model.")

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

MODEL_DIR = Path("models")

model_files = list(MODEL_DIR.glob("*.pkl"))

if not model_files:
    st.error("No trained model found in models folder.")
    st.stop()

latest_model = sorted(model_files)[-1]

model = joblib.load(latest_model)

st.success(f"Loaded model: {latest_model.name}")

# ---------------------------------------------------
# GET FEATURE NAMES FROM MODEL
# ---------------------------------------------------

try:
    feature_names = model.get_booster().feature_names
except:
    st.error("Could not read feature names from model.")
    st.stop()

# ---------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------

st.sidebar.header("Input Features")

hour = st.sidebar.slider("Hour of Day", 0, 23, 12)

day_name = st.sidebar.selectbox(
    "Day of Week",
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
)

month = st.sidebar.slider("Month", 1, 12, 6)

is_weekend = st.sidebar.selectbox("Weekend?", [0,1])

day_map = {
    "Monday":0,
    "Tuesday":1,
    "Wednesday":2,
    "Thursday":3,
    "Friday":4,
    "Saturday":5,
    "Sunday":6
}

day_of_week = day_map[day_name]

# ---------------------------------------------------
# BUILD FEATURE DATAFRAME
# ---------------------------------------------------

features = pd.DataFrame(
    np.zeros((1, len(feature_names))),
    columns=feature_names
)

# base inputs
if "hour" in features.columns:
    features.loc[0, "hour"] = hour

if "dow" in features.columns:
    features.loc[0, "dow"] = day_of_week

if "month" in features.columns:
    features.loc[0, "month"] = month

if "is_weekend" in features.columns:
    features.loc[0, "is_weekend"] = is_weekend

if "year" in features.columns:
    features.loc[0, "year"] = 2022

# ---------------------------------------------------
# CYCLIC FEATURES
# ---------------------------------------------------

if "hour_sin" in features.columns:
    features.loc[0, "hour_sin"] = math.sin(2 * math.pi * hour / 24)

if "hour_cos" in features.columns:
    features.loc[0, "hour_cos"] = math.cos(2 * math.pi * hour / 24)

if "dow_sin" in features.columns:
    features.loc[0, "dow_sin"] = math.sin(2 * math.pi * day_of_week / 7)

if "dow_cos" in features.columns:
    features.loc[0, "dow_cos"] = math.cos(2 * math.pi * day_of_week / 7)

# ---------------------------------------------------
# RUSH HOUR FEATURES
# ---------------------------------------------------

if "is_morning_rush" in features.columns:
    features.loc[0, "is_morning_rush"] = 1 if 7 <= hour <= 10 else 0

if "is_evening_rush" in features.columns:
    features.loc[0, "is_evening_rush"] = 1 if 17 <= hour <= 20 else 0

if "is_night" in features.columns:
    features.loc[0, "is_night"] = 1 if hour >= 22 or hour <= 5 else 0

# ---------------------------------------------------
# DISPLAY MODEL INPUT
# ---------------------------------------------------

st.subheader("Model Input")
st.dataframe(features)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if st.button("Predict Demand"):

    try:
        prediction = model.predict(features)[0]

        st.subheader("Prediction Result")

        st.metric(
            label="Predicted Ride Demand",
            value=int(prediction)
        )

    except Exception as e:
        st.error("Prediction failed")
        st.write(str(e))

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")
st.write("OLA Demand Forecasting Dashboard")
