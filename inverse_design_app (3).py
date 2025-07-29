import streamlit as st
import numpy as np
import joblib
from scipy.optimize import differential_evolution

# --------------------------
# Load trained model & scalers
# --------------------------
model = joblib.load("scgpc_model_bundle.pkl")
input_scaler = joblib.load("input_scaler.pkl")
output_scaler = joblib.load("output_scaler.pkl")

# --------------------------
# Feature Names & Bounds
# --------------------------
feature_names = [
    "Fly Ash", "GGBS", "NaOH", "Molarity", "Silicate Solution",
    "Sand", "Coarse Agg", "Water", "SP", "Temperature"
]

bounds = [
    (300, 600),   # Fly Ash
    (50, 300),    # GGBS
    (5, 50),      # NaOH
    (8, 16),      # Molarity
    (100, 300),   # Silicate Solution
    (600, 900),   # Sand
    (700, 1200),  # Coarse Aggregate
    (120, 220),   # Water
    (0, 15),      # SP
    (20, 80)      # Temperature
]

# --------------------------
# Inverse Design Function
# --------------------------
def inverse_design(target_output, bounds, max_iter=100):
    scaled_target = output_scaler.transform([target_output])[0]

    def fitness_function(input_array):
        input_array = np.array(input_array).reshape(1, -1)
        scaled_input = input_scaler.transform(input_array)
        scaled_pred = model.predict(scaled_input)[0]
        mse = np.mean((scaled_pred - scaled_target)**2)
        return mse

    result = differential_evolution(fitness_function, bounds, maxiter=max_iter, seed=42)
    optimized_input = result.x.reshape(1, -1)
    scaled_input = input_scaler.transform(optimized_input)
    scaled_output = model.predict(scaled_input)
    predicted_output = output_scaler.inverse_transform(scaled_output)[0]
    return optimized_input.flatten(), predicted_output

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="SCGPC Inverse Design", layout="centered")
st.title("🧪 Inverse Design of Self-Compacting Geopolymer Concrete Mix")

st.markdown("### 🎯 Enter Desired Target Properties:")
cs28 = st.number_input("Compressive Strength (MPa)", min_value=10.0, max_value=100.0, value=45.0)
sf = st.number_input("Slump Flow (mm)", min_value=400.0, max_value=800.0, value=650.0)
t500 = st.number_input("Flow Time (T500 sec)", min_value=1.0, max_value=10.0, value=2.5)

if st.button("Suggest Mix Design"):
    with st.spinner("Running optimization..."):
        optimized_mix, predicted_props = inverse_design([cs28, sf, t500], bounds)

    st.subheader("🧱 Suggested Mix Proportions:")
    for name, val in zip(feature_names, optimized_mix):
        st.write(f"**{name}:** {val:.2f}")

    st.subheader("📈 Predicted Properties:")
    st.write(f"**CS28:** {predicted_props[0]:.2f} MPa")
    st.write(f"**Slump Flow:** {predicted_props[1]:.2f} mm")
    st.write(f"**T500:** {predicted_props[2]:.2f} sec")
