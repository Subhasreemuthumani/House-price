import pandas as pd
import joblib

# Load trained model
try:
    model = joblib.load("house_model.pkl")
except FileNotFoundError:
    print("❌ Model file not found! Train it using train_model.py")
    exit()

# Feature order
features = ["Gr Liv Area", "Overall Qual", "Garage Cars", "Total Bsmt SF"]

# Get user input
print("🔢 Enter house details:")
input_values = []
for feat in features:
    val = float(input(f"{feat}: "))
    input_values.append(val)

# Create DataFrame for prediction
X_input = pd.DataFrame([input_values], columns=features)

# Predict price
predicted_price = model.predict(X_input)[0]
print(f"💰 Predicted House Price: ₹{predicted_price:,.2f}")
