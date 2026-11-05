import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("temperature.csv", sep="\t")

# Create numerical input
df["Day"] = range(1, len(df) + 1)

# Input and output
X = df[["Day"]]
y = df["Temperature"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "temperature_model.pkl")

print("Model saved successfully!")