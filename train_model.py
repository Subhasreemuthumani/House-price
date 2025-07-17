import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load cleaned dataset
df = pd.read_csv("data/cleaned_house.csv")

# Updated feature list with an additional feature
features = ['Gr Liv Area', 'Overall Qual', 'Garage Cars', 'Total Bsmt SF', 'YearBuilt']

# Ensure all required features exist
missing = [f for f in features if f not in df.columns]
if missing:
    raise ValueError(f"Missing features in dataset: {missing}")

X = df[features]
y = df["SalePrice"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/house_model.pkl")
print("✅ Model trained and saved successfully.")
