import streamlit as st
import numpy as np
import joblib

# Load model and scalers from a single bundled file
model_bundle = joblib.load("scgpc_model_bundle.pkl")
model = model_bundle["model"]
input_scaler = model_bundle["input_scaler"]
output_scaler = model_bundle["output_scaler"]

# Inverse design bounds
bounds = [
    (300, 500),    # Fly Ash
    (100, 250),    # GGBS
    (10, 60),      # NaOH
    (8, 16),       # Molarity
    (100, 300),    # Silicate Soln
    (650, 800),    # Sand
    (850, 1100),   # Coarse Agg
    (140, 220),    # Water
    (1.0, 3.5),    # SP
    (25, 85)       # Temperature
]

# Define inverse design function using Genetic Algorithm (simplified demo)
def inverse_design(target_outputs, bounds):
    from scipy.optimize import differential_evolution

    def objective(x):
        x_scaled = input_scaler.transform([x])
        y_pred_scaled = model.predict(x_scaled)
        y_pred = output_scaler.inverse_transform(y_pred_scaled)
        return np.linalg.norm(np.array(y_pred[0]) - np.array(target_outputs))

    result = differential_evolution(objective, bounds, maxiter=100, popsize=15)
    best_mix = result.x
    best_pred = output_scaler.inverse_transform(model.predict(input_scaler.transform([best_mix])))[0]
    return best_mix, best_pred

# Streamlit App
st.title("Inverse Design of SCGPC Mix")

st.markdown("### Input Desired Concrete Properties (Realistic Ranges)")
cs = st.number_input("Compressive Strength (MPa) [20-80]", min_value=20.0, max_value=80.0, value=40.0)
sf = st.number_input("Slump Flow (mm) [400-800]", min_value=400.0, max_value=800.0, value=650.0)
t500 = st.number_input("Flow Time T500 (sec) [1-10]", min_value=1.0, max_value=10.0, value=3.5)

if st.button("Suggest Mix Design", key="suggest_button"):
    clipped_cs = np.clip(cs, 20.0, 80.0)
    clipped_sf = np.clip(sf, 400.0, 800.0)
    clipped_t500 = np.clip(t500, 1.0, 10.0)
    
    mix, predicted = inverse_design([clipped_cs, clipped_sf, clipped_t500], bounds)
    
    st.subheader("Suggested Mix Design Proportions")
    labels = ["Fly Ash", "GGBS", "NaOH", "Molarity", "Silicate Soln", "Sand", "Coarse Agg", "Water", "SP", "Temp"]
    for name, val in zip(labels, mix):
        st.write(f"{name}: {val:.2f}")

    st.subheader("Predicted Properties of Suggested Mix")
    st.write(f"Compressive Strength (CS28): {predicted[0]:.2f} MPa")
    st.write(f"Slump Flow (SF): {predicted[1]:.2f} mm")
    st.write(f"Flow Time (T500): {predicted[2]:.2f} sec")
