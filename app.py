import streamlit as st
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

import pandas as pd
import joblib

# Currency formatter
def format_in_inr(amount):
    return f"₹{int(amount):,}"

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load("house_model.pkl")
    except FileNotFoundError:
        st.error("❌ Model file not found in models/house_model.pkl")
        return None

model = load_model()

# App title
st.title("🏠 House Price Prediction (in ₹)")

# Sidebar inputs
st.sidebar.header("📋 Enter House Features")
GrLivArea = st.sidebar.number_input("🏠 Gr Liv Area (sqft)", min_value=0, value=1500)
OverallQual = st.sidebar.slider("📊 Overall Quality (1-10)", 1, 10, 5)
GarageCars = st.sidebar.slider("🚗 Garage Capacity", 0, 5, 2)
TotalBsmtSF = st.sidebar.number_input("🏚 Total Basement SF", min_value=0, value=800)
YearBuilt = st.sidebar.number_input("📆 Year Built", min_value=1800, max_value=2025, value=2000)

# Predict
if st.sidebar.button("🔍 Predict Price"):
    if model:
        input_data = pd.DataFrame([[GrLivArea, OverallQual, GarageCars, TotalBsmtSF, YearBuilt]],
                                  columns=["Gr Liv Area", "Overall Qual", "Garage Cars", "Total Bsmt SF", "Year Built"])
        try:
            prediction = model.predict(input_data)[0]
            st.success(f"💰 Estimated House Price: **{format_in_inr(prediction)}**")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
    else:
        st.warning("⚠️ No model loaded.")
