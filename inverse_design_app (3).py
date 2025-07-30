
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import differential_evolution

# Load model
model = joblib.load("best_model.pkl")

# Define bounds for each input feature (adjust according to your dataset)
feature_bounds = {
    "Fly Ash": (300, 500),
    "GGBS": (50, 200),
    "NaOH": (10, 25),
    "Molarity": (8, 16),
    "Silicate Solution": (100, 250),
    "Sand": (600, 800),
    "Coarse Agg": (900, 1000),
    "Water": (150, 220),
    "SP": (0.5, 5.0),
    "Temperature": (25, 90)
}

feature_names = list(feature_bounds.keys())
bounds = list(feature_bounds.values())

# Streamlit UI
st.title("🧱 Inverse Design of SCGPC Mix")
st.write("Enter target concrete properties to generate suitable mix design.")

# Get target values from user input
target_cs = st.number_input("Target Compressive Strength (MPa)", min_value=0.0, value=45.0)
target_sf = st.number_input("Target Slump Flow (mm)", min_value=0.0, value=650.0)
target_t500 = st.number_input("Target T500 Flow Time (s)", min_value=0.0, value=3.5)

# Prepare target as array
target_real = np.array([[target_cs, target_sf, target_t500]])

# Define objective function
def objective_function(x):
    x = np.array(x).reshape(1, -1)
    y_pred = model.predict(x)
    loss = np.linalg.norm(y_pred - target_real.flatten())
    return loss



if st.button("🔍 Run Inverse Design"):
    with st.spinner("Running inverse optimization..."):
        result = differential_evolution(objective_function, bounds, seed=42)
        best_mix = result.x  # This is a 1D array of optimal feature values

        predicted = model.predict(best_mix)[0]

        print("🎯 Predicted Concrete Properties:")
        print(f"Compressive Strength: {predicted[0]:.2f} MPa")
        print(f"Slump Flow: {predicted[1]:.2f} mm")
        print(f"T500: {predicted[2]:.2f} sec")

        print("🧱 Suggested Mix Design:")
        print(best_mix.flatten())


        st.subheader("🧱 Suggested Mix Design")
        mix_dict = {name: round(val, 2) for name, val in zip(feature_names, best_mix)}
        st.dataframe(pd.DataFrame(mix_dict.items(), columns=["Component", "Value"]))
