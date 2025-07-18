import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load cleaned data
df = pd.read_csv("cleaned_house.csv")

# Select features and target
features = ["Gr Liv Area", "Overall Qual", "Garage Cars", "Total Bsmt SF"]
target = "SalePrice"

X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "house_model.pkl")
print("✅ Model trained and saved as house_model.pkl")
