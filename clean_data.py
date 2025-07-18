import pandas as pd

# Load dataset
df = pd.read_csv("house.csv")

# Drop columns with >30% missing values
df = df.dropna(thresh=len(df) * 0.7, axis=1)

# Fill numeric NaNs with median
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical NaNs with mode
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Drop 'Id' column if present
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# Save cleaned dataset
df.to_csv("cleaned_house.csv", index=False)
print("✅ Cleaned data saved to 'cleaned_house.csv'")
