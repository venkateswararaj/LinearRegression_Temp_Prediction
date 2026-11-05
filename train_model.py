import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Read dataset
df = pd.read_csv("temperature.csv", sep="\t")

# 2. Convert date
df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y"
)

# 3. Create numerical input
df["Day"] = range(1, len(df) + 1)

# 4. Input and output
X = df[["Day"]]
y = df["Temperature"]

# 5. Train model
model = LinearRegression()
model.fit(X, y)

# 6. Predict on training data
predictions = model.predict(X)

# 7. Evaluate model
print("MAE:", mean_absolute_error(y, predictions))
print("R2 Score:", r2_score(y, predictions))

# 8. Predict new temperature
new_day = pd.DataFrame({"Day": [5]})
prediction = model.predict(new_day)

print("Predicted temperature for Day 5:", prediction[0])

# 9. Visualize graph
plt.scatter(X, y, label="Actual Temperature")
plt.plot(X, predictions, color="red", label="Regression Line")

plt.xlabel("Day")
plt.ylabel("Temperature (Celsius)")
plt.title("Temperature Prediction - Linear Regression")
plt.legend()
plt.grid()
plt.show()