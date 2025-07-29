
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize

# Load the model and scalers
model_bundle = joblib.load("scgpc_model_bundle.pkl")
model = model_bundle["model"]
input_scaler = model_bundle["input_scaler"]
output_scaler = model_bundle["output_scaler"]

st.title("Inverse Design of Self-Compacting Geopolymer Concrete")

st.sidebar.header("Target Properties")
cs_target = st.sidebar.number_input("Compressive Strength (CS28)", min_value=0.0, max_value=100.0, value=40.0)
sf_target = st.sidebar.number_input("Slump Flow (SF)", min_value=0.0, max_value=1000.0, value=700.0)
t500_target = st.sidebar.number_input("T500 Flow Time (T500)", min_value=0.0, max_value=100.0, value=5.0)

target = np.array([[cs_target, sf_target, t500_target]])

# Inverse design objective function
def objective(x):
    x_scaled = input_scaler.transform([x])
    y_pred_scaled = model.predict(x_scaled)
    y_pred = output_scaler.inverse_transform(y_pred_scaled)
    loss = np.mean((y_pred - target)**2)
    return loss

# Generate initial guess
try:
    x0 = input_scaler.mean_  # For StandardScaler
except AttributeError:
    x0 = np.ones(input_scaler.n_features_in_) * 100  # Fallback

from scipy.optimize import differential_evolution

# Define realistic bounds
bounds = [
    (300, 500),     # Fly Ash
    (50, 250),      # GGBS
    (10, 60),       # NaOH
    (8, 16),        # Molarity
    (100, 250),     # Silicate Soln
    (600, 900),     # Sand
    (800, 1100),    # Coarse Agg
    (150, 250),     # Water
    (1, 5),         # SP
    (25, 90)        # Temp
]

# Define loss function
def loss_fn(x):
    x_scaled = input_scaler.transform([x])
    y_pred_scaled = model.predict(x_scaled)
    y_pred = output_scaler.inverse_transform(y_pred_scaled)
    loss = np.sum((y_pred - target) ** 2)
    return loss





if st.button("Run Inverse Design"):
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    if result.success:
        optimal_input = result.x
        input_names = ["Fly Ash", "GGBS", "NaOH", "Molarity", "Silicate Soln", "Sand", "Coarse Agg", "Water", "SP", "Temp"]
        output_df = pd.DataFrame([optimal_input], columns=input_names)
        st.success("Optimized Mix Design Found!")
        st.dataframe(output_df.style.format("{:.2f}"))
    else:

        st.error("Optimization failed.")
# Use differential evolution instead of minimize
result = differential_evolution(loss_fn, bounds, strategy='best1bin', maxiter=200, popsize=20, tol=1e-4)

# Final optimized input
suggested_mix = result.x
