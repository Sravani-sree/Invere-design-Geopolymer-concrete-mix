# inverse_design_app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize

# Load the trained model bundle
bundle = joblib.load("scgpc_model_bundle.pkl")
model = bundle["model"]
input_scaler = bundle["input_scaler"]
output_scaler = bundle["output_scaler"]

# Define inverse design function
def inverse_design(target_cs, target_sf, target_t500):
    target = np.array([[target_cs, target_sf, target_t500]])
    y_target_scaled = output_scaler.transform(target)

    def objective(x):
        x_scaled = input_scaler.transform([x])
        y_pred_scaled = model.predict(x_scaled)
        loss = np.mean((y_pred_scaled - y_target_scaled) ** 2)
        return loss

    # Initial guess and bounds
    x0 = np.mean(input_scaler.data_, axis=0)
    bounds = [(0.1, 1000)] * len(x0)

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

    if result.success:
        return result.x
    else:
        return None

# Streamlit UI
st.title("🧪 Inverse Design: SCGPC Mix Optimization")

st.sidebar.header("🎯 Desired Performance Targets")
target_cs = st.sidebar.number_input("Compressive Strength (CS28)", min_value=0.0, max_value=100.0, value=30.0)
target_sf = st.sidebar.number_input("Slump Flow (SF)", min_value=100.0, max_value=1000.0, value=600.0)
target_t500 = st.sidebar.number_input("T500 Flow Time", min_value=0.0, max_value=50.0, value=6.0)

if st.sidebar.button("🔍 Optimize Mix Design"):
    optimized_mix = inverse_design(target_cs, target_sf, target_t500)

    if optimized_mix is not None:
        input_columns = [
            'Fly Ash', 'GGBS', 'NaOH', 'Molarity',
            'Silicate solution', 'sand', 'coarse aggregates',
            'water', 'Spz', 'Temperature'
        ]
        result_df = pd.DataFrame([optimized_mix], columns=input_columns)

        st.success("✅ Optimized Mix Design Found!")
        st.dataframe(result_df.style.format(precision=2))

        st.bar_chart(result_df.T.rename(columns={0: "Mix Proportion"}))

    else:
        st.error("❌ Optimization failed. Try different target values.")

st.markdown("---")
st.caption("Developed for SCGPC Inverse Design | Pattupogula Sravani")
