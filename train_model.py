import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load cleaned data
df = pd.read_csv("data/cleaned_house.csv")

# Check that 'Year Built' is present
assert 'Year Built' in df.columns, "❌ 'Year Built' column not found in dataset!"

# Define features and target
features = ['Gr Liv Area', 'Overall Qual', 'Garage Cars', 'Total Bsmt SF', 'Year Built']
X = df[features]
y = df['SalePrice']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "house_model.pkl")
print("✅ Model trained and saved to models/house_model.pkl")
