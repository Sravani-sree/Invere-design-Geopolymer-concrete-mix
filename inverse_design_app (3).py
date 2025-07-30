
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import differential_evolution

# Load model
model = joblib.load("best_model.pkl")

# Define bounds for each input feature (adjust according to your dataset)
bounds = [
    (200, 500),   # Fly Ash
    (50, 300),    # GGBS
    (10, 50),     # NaOH
    (8, 16),      # Molarity
    (100, 250),   # Silicate
    (500, 800),   # Sand
    (800, 1100),  # Coarse Aggregate
    (100, 250),   # Water
    (0.5, 5),     # SP
    (25, 85)      # Curing Temp
]
feature_names = list(feature_bounds.keys())
bounds = list(feature_bounds.values())

# Streamlit UI
st.title("🧱 Inverse Design of SCGPC Mix")
st.write("Enter target concrete properties to generate suitable mix design.")

target_cs = st.number_input("🎯 Target Compressive Strength (MPa)", min_value=10.0, max_value=100.0, value=45.0)
target_sf = st.number_input("🎯 Target Slump Flow (mm)", min_value=400.0, max_value=800.0, value=650.0)
target_t500 = st.number_input("🎯 Target T500 Flow Time (s)", min_value=0.5, max_value=10.0, value=3.5)

target_values = np.array([target_cs, target_sf, target_t500])

def objective_function(x):
    x_reshaped = np.array(x).reshape(1, -1)
    y_pred = model.predict(x_reshaped)[0]
    loss = np.linalg.norm(y_pred - target_real.flatten())  # target_real is also in real-world units
    return loss


if st.button("🔍 Run Inverse Design"):
    with st.spinner("Running inverse optimization..."):
        result = differential_evolution(objective_function, bounds)
        best_mix = result.x.reshape(1, -1)
        predicted = model.predict(best_mix)[0]

        print("🎯 Predicted Concrete Properties:")
        print(f"Compressive Strength: {predicted[0]:.2f} MPa")
        print(f"Slump Flow: {predicted[1]:.2f} mm")
        print(f"T500: {predicted[2]:.2f} sec")

        print("🧱 Suggested Mix Design:")
        print(best_mix.flatten())

        )

        st.subheader("🧱 Suggested Mix Design")
        mix_dict = {name: round(val, 2) for name, val in zip(feature_names, best_mix)}
        st.dataframe(pd.DataFrame(mix_dict.items(), columns=["Component", "Value"]))
