
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="G03 Crop Yield Predictor",
    page_icon="🌾",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "G03_model.joblib"
CONFIG_PATH = BASE_DIR / "models" / "G03_app_config.json"

st.title("G03 Crop Yield Prediction")
st.subheader("CSC 4110 — Machine Learning Group Project")
st.write("**Group:** G03 — Crop Yield")
st.write(
    "**Group members:** Brayson Imbambi, Simon Njoroge, "
    "Jacob Chemning’wa, Austin Gatita, Agnes Watti"
)
st.write(
    "**Dataset:** FAO Crop Yield Dataset (`yield_df.csv`), "
    "cleaned as `clean_crops.csv`."
)

if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
    st.error("The saved model or application configuration could not be found.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

model = load_model()
config = load_config()

feature_cols = config["feature_cols"]
item_dummy_cols = config["item_dummy_cols"]
item_options = config["item_options"]
area_frequency_mapping = config["area_frequency_mapping"]
environment_defaults = config["environment_defaults"]

st.markdown("### Enter crop and location details")

selected_item = st.selectbox("Item (crop)", item_options)
area_options = sorted(area_frequency_mapping.keys())
selected_area = st.selectbox("Area (country)", area_options)

year = st.number_input(
    "Year",
    min_value=1990,
    max_value=2035,
    value=2023,
    step=1
)

st.markdown("### Environmental information")

rain_mm = st.number_input(
    "Annual rainfall (mm)",
    min_value=0.0,
    value=float(environment_defaults["rain_mm"]),
    step=10.0
)

pesticides_t = st.number_input(
    "Pesticides (tonnes)",
    min_value=0.0,
    value=float(environment_defaults["pesticides_t"]),
    step=10.0
)

avg_temp = st.number_input(
    "Average temperature (°C)",
    value=float(environment_defaults["avg_temp"]),
    step=0.1
)

if st.button("Predict crop yield", type="primary"):
    # Create one row with the same feature names and order used during training.
    row = {feature: 0.0 for feature in feature_cols}

    # Set the crop one-hot encoded feature.
    selected_dummy = "Item_" + selected_item
    if selected_dummy in row:
        row[selected_dummy] = 1.0

    # Set the country frequency calculated from the original Task 2.2 data.
    row["Area_frequency"] = float(area_frequency_mapping[selected_area])

    # Set the year-based and environmental features.
    if "years_since_1990" in row:
        row["years_since_1990"] = float(year - 1990)

    if "rain_mm" in row:
        row["rain_mm"] = float(rain_mm)

    if "pesticides_t" in row:
        row["pesticides_t"] = float(pesticides_t)

    if "avg_temp" in row:
        row["avg_temp"] = float(avg_temp)

    input_df = pd.DataFrame([row], columns=feature_cols)

    try:
        predicted_log_yield = float(model.predict(input_df)[0])

        # The group created log_yield using the natural logarithm.
        predicted_hg_ha = float(np.exp(predicted_log_yield))
        predicted_tonnes_ha = predicted_hg_ha / 10000.0

        st.success(f"Prediction completed for {selected_item} in {selected_area}.")
        st.metric("Estimated crop yield", f"{predicted_hg_ha:,.2f} hg/ha")
        st.metric("Estimated crop yield", f"{predicted_tonnes_ha:,.4f} tonnes/ha")
        st.caption(
            "This is a statistical estimate based on the group's trained model, "
            "not a guaranteed agricultural outcome."
        )
    except Exception as error:
        st.error(f"Prediction failed: {error}")
