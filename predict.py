import pandas as pd
import joblib
import locale

# Format for Indian currency
locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')

# Load trained model
model = joblib.load("models/house_model.pkl")

features = ['Gr Liv Area', 'Overall Qual', 'Garage Cars', 'Total Bsmt SF', 'YearBuilt']
input_values = []

print("🔢 Enter house details:")
for feat in features:
    val = float(input(f"{feat}: "))
    input_values.append(val)

X_input = pd.DataFrame([input_values], columns=features)
predicted_price = model.predict(X_input)[0]

formatted = locale.currency(predicted_price, grouping=True)
print(f"💰 Predicted House Price: {formatted}")
