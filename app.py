import streamlit as st
import pandas as pd
import joblib
import locale

# Format for Indian currency
locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')
def load_model():
    return joblib.load("models/house_model.pkl")

model = load_model()

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor (India)")

st.sidebar.header("📋 Enter House Details")
GrLivArea = st.sidebar.number_input("Gr Liv Area (sqft)", min_value=0, value=1500)
OverallQual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 6)
GarageCars = st.sidebar.slider("Garage Capacity", 0, 5, 2)
TotalBsmtSF = st.sidebar.number_input("Total Basement SF", min_value=0, value=800)
YearBuilt = st.sidebar.number_input("Year Built", min_value=1800, max_value=2025, value=2000)

features = ['Gr Liv Area', 'Overall Qual', 'Garage Cars', 'Total Bsmt SF', 'YearBuilt']
input_df = pd.DataFrame([[GrLivArea, OverallQual, GarageCars, TotalBsmtSF, YearBuilt]], columns=features)

if st.sidebar.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    formatted_price = locale.currency(prediction, grouping=True)
    st.success(f"💰 Estimated Price: **{formatted_price}**")
