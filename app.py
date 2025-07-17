import streamlit as st
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")  # Must be first Streamlit command

import pandas as pd
import joblib

# Format as Indian Rupees
def format_in_inr(amount):
    return f"₹{int(amount):,}"

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/house_model.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found! Please train and save the model in models/house_model.pkl.")
        return None

model = load_model()

# App title
st.title("🏠 House Price Prediction (in ₹)")

# Sidebar inputs
st.sidebar.header("📋 Enter House Features")
GrLivArea = st.sidebar.number_input("🏠 Gr Liv Area (sqft)", min_value=0, value=1500)
OverallQual = st.sidebar.slider("📊 Overall Quality (1-10)", 1, 10, 5)
GarageCars = st.sidebar.slider("🚗 Garage Capacity (Cars)", 0, 5, 2)
TotalBsmtSF = st.sidebar.number_input("🏚 Total Basement SF", min_value=0, value=800)

# Predict
if st.sidebar.button("🔍 Predict Price"):
    if model:
        input_data = pd.DataFrame([[GrLivArea, OverallQual, GarageCars, TotalBsmtSF]],
                                  columns=["Gr Liv Area", "Overall Qual", "Garage Cars", "Total Bsmt SF"])
        try:
            prediction = model.predict(input_data)[0]
            formatted_price = format_in_inr(prediction)
            st.success(f"💰 Estimated House Price: **{formatted_price}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.warning("⚠️ Model not loaded. Cannot make predictions.")
