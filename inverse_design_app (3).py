import streamlit as st
import numpy as np
import joblib
from scipy.optimize import differential_evolution

# Load model and scalers
bundle = joblib.load("scgpc_model_bundle.pkl")
model = bundle["model"]
input_scaler = bundle["input_scaler"]
output_scaler = bundle["output_scaler"]

st.title("Inverse Design: Self-Compacting Geopolymer Concrete")

st.markdown("### Enter Desired Target Properties:")
cs_target = st.number_input("Compressive Strength (CS28) in MPa", min_value=20.0, max_value=80.0, value=40.0)
sf_target = st.number_input("Slump Flow (SF) in mm", min_value=400.0, max_value=800.0, value=600.0)
t500_target = st.number_input("T500 Flow Time in sec", min_value=1.0, max_value=10.0, value=5.0)

target_output = np.array([cs_target, sf_target, t500_target])
# Clip to avoid extreme values
target_output = np.clip(target_output, [20, 400, 1], [80, 800, 10])

# Define realistic bounds for each component (order must match training data)
bounds = [
    (300, 600),   # Fly Ash
    (50, 300),    # GGBS
    (10, 60),     # NaOH
    (8, 16),      # Molarity
    (100, 300),   # Silicate
    (500, 900),   # Sand
    (700, 1100),  # Coarse Aggregate
    (100, 250),   # Water
    (0.5, 5),     # SP
    (20, 90)      # Temperature
]

# Define objective function
def objective(x):
    x_scaled = input_scaler.transform([x])
    y_pred_scaled = model.predict(x_scaled)
    y_pred = output_scaler.inverse_transform(y_pred_scaled)
    return np.sum((y_pred[0] - target_output)**2)

if st.button("Suggest Mix Design"):
    result = differential_evolution(objective, bounds, maxiter=1000, popsize=15, seed=42)

    if result.success:
        suggested_mix = result.x
        predicted_output = output_scaler.inverse_transform(model.predict(input_scaler.transform([suggested_mix])))[0]

        st.subheader("Suggested Mix Design Proportions:")
        labels = [
            "Fly Ash (kg/m³)", "GGBS (kg/m³)", "NaOH (kg/m³)", "NaOH Molarity",
            "Sodium Silicate (kg/m³)", "Sand (kg/m³)", "Coarse Aggregate (kg/m³)",
            "Water (kg/m³)", "Superplasticizer (kg/m³)", "Curing Temperature (°C)"
        ]
        for label, value in zip(labels, suggested_mix):
            st.write(f"**{label}:** {value:.2f}")

        st.subheader("Predicted Performance Properties:")
        st.write(f"**Compressive Strength (CS28):** {predicted_output[0]:.2f} MPa")
        st.write(f"**Slump Flow (SF):** {predicted_output[1]:.2f} mm")
        st.write(f"**T500 Flow Time:** {predicted_output[2]:.2f} sec")
    else:
        st.error("Optimization failed. Please try different targets.")
