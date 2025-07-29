
import streamlit as st
import numpy as np
import joblib
from scipy.optimize import differential_evolution

# Load model and scalers
model_data = joblib.load("scgpc_model_bundle.pkl")
model = model_data["model"]
input_scaler = model_data["input_scaler"]
output_scaler = model_data["output_scaler"]

st.title("Inverse Design of Self-Compacting Geopolymer Concrete Mix")

# Sidebar: Target properties
st.sidebar.header("Set Desired Performance Properties")
target_CS28 = st.sidebar.slider("Compressive Strength (CS28) [MPa]", 10.0, 80.0, 40.0)
target_SF = st.sidebar.slider("Slump Flow (SF) [mm]", 500.0, 800.0, 700.0)
target_T500 = st.sidebar.slider("Flow Time (T500) [sec]", 2.0, 20.0, 11.0)
target = np.array([[target_CS28, target_SF, target_T500]])

# Realistic bounds for each mix component
bounds = [
    (300, 500),     # Fly Ash (kg/m³)
    (50, 250),      # GGBS (kg/m³)
    (10, 60),       # NaOH (kg/m³)
    (8, 16),        # Molarity (M)
    (100, 250),     # Sodium Silicate Solution (kg/m³)
    (600, 900),     # Sand (kg/m³)
    (800, 1100),    # Coarse Aggregate (kg/m³)
    (150, 250),     # Water (kg/m³)
    (1, 5),         # Superplasticizer (%)
    (25, 90)        # Temperature (°C)
]

# Show bounds to user
with st.expander("🔍 View Realistic Bounds for Each Input Feature"):
    st.markdown("""
    | Feature               | Lower Bound | Upper Bound |
    |------------------------|-------------|-------------|
    | Fly Ash (kg/m³)        | 300         | 500         |
    | GGBS (kg/m³)           | 50          | 250         |
    | NaOH (kg/m³)           | 10          | 60          |
    | Molarity (M)           | 8           | 16          |
    | Sodium Silicate (kg/m³)| 100         | 250         |
    | Sand (kg/m³)           | 600         | 900         |
    | Coarse Aggregate (kg/m³)| 800        | 1100        |
    | Water (kg/m³)          | 150         | 250         |
    | Superplasticizer (%)   | 1           | 5           |
    | Temperature (°C)       | 25          | 90          |
    """)

# Optimization loss function
def loss_fn(x):
    x_scaled = input_scaler.transform([x])
    y_pred_scaled = model.predict(x_scaled)
    y_pred = output_scaler.inverse_transform(y_pred_scaled)
    loss = np.sum((y_pred - target) ** 2)
    return loss

# Run optimizer
if st.button("Suggest Mix Design"):
    with st.spinner("Optimizing..."):
        result = differential_evolution(loss_fn, bounds, strategy='best1bin', maxiter=200, popsize=20, tol=1e-4, seed=42)
        suggested_mix = result.x

        # Predict target performance from optimized input
        optimized_input_scaled = input_scaler.transform([suggested_mix])
        predicted_output_scaled = model.predict(optimized_input_scaled)
        predicted_properties = output_scaler.inverse_transform(predicted_output_scaled)[0]
        predicted_CS28, predicted_SF, predicted_T500 = predicted_properties

        # Show mix design
        st.success("Optimized Mix Design Found:")
        labels = ["Fly Ash", "GGBS", "NaOH", "Molarity", "Silicate Soln", "Sand", "Coarse Agg", "Water", "SP", "Temp"]
        for i, val in enumerate(suggested_mix):
            st.markdown(f"- **{labels[i]}:** {val:.2f}")

        st.subheader("Predicted Properties of Suggested Mix:")
        st.markdown(f"- **Compressive Strength (CS28):** {predicted_CS28:.2f} MPa")
        st.markdown(f"- **Slump Flow (SF):** {predicted_SF:.2f} mm")
        st.markdown(f"- **Flow Time (T500):** {predicted_T500:.2f} sec")
