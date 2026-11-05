import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.title("Temperature Prediction")
st.write("Linear Regression Machine Learning Project")

# Load dataset
df = pd.read_csv("temperature.csv", sep="\t")

# Load trained model
model = joblib.load("temperature_model.pkl")

# Prepare dataset
df["Day"] = range(1, len(df) + 1)

# Display dataset
st.subheader("Temperature Dataset")
st.dataframe(df)

# User input
st.subheader("Predict Temperature")

day = st.number_input(
    "Enter Day Number",
    min_value=1,
    max_value=365,
    value=5
)

# Prediction
if st.button("Predict Temperature"):
    input_data = pd.DataFrame({"Day": [day]})
    prediction = model.predict(input_data)

    st.success(
        f"Predicted Temperature: {prediction[0]:.2f} °C"
    )

# Graph
st.subheader("Temperature Graph")

fig, ax = plt.subplots()

X = df[["Day"]]
y = df["Temperature"]

ax.scatter(X, y, label="Actual Temperature")
ax.plot(X, model.predict(X), color="red",
        label="Regression Line")

ax.set_xlabel("Day")
ax.set_ylabel("Temperature (°C)")
ax.legend()

st.pyplot(fig)