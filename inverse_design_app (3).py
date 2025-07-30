
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import differential_evolution

# Load model
model = joblib.load("best_model.pkl")

# Define bounds for each input feature (adjust according to your dataset)
feature_bounds = {
    "Fly Ash": (300, 600),
    "GGBS": (100, 300),
    "NaOH": (10, 60),
    "Molarity": (8, 16),
    "Sodium Silicate": (100, 250),
    "Sand": (600, 900),
    "Coarse Agg": (800, 1100),
    "Water": (150, 250),
    "Superplasticizer": (0.5, 3.5),
    "Curing Temp": (25, 90)
}
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
    x = np.array(x).reshape(1, -1)
    prediction = model.predict(x)[0]
    error = np.mean((prediction - target_values) ** 2)
    return error

if st.button("🔍 Run Inverse Design"):
    with st.spinner("Running inverse optimization..."):
        result = differential_evolution(objective_function, bounds, seed=42, maxiter=200, tol=1e-3)
        best_mix = result.x
        predicted_props = model.predict([best_mix])[0]

        st.success("Mix design successfully generated!")

        st.subheader("🎯 Predicted Properties (from optimized mix)")
        st.json({
            "C Strength": round(predicted_props[0], 2),
            "S flow": round(predicted_props[1], 2),
            "T 500": round(predicted_props[2], 2)
        })

        st.subheader("🧱 Suggested Mix Design")
        mix_dict = {name: round(val, 2) for name, val in zip(feature_names, best_mix)}
        st.dataframe(pd.DataFrame(mix_dict.items(), columns=["Component", "Value"]))
