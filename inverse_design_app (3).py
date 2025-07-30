import streamlit as st
import numpy as np
import joblib
from scipy.optimize import differential_evolution

# Load trained model
model = joblib.load("best_model.pkl")

# Feature bounds (in real-world units) – update according to your dataset
feature_bounds = {
    "Fly Ash": (300, 600),
    "GGBS": (50, 300),
    "NaOH": (10, 30),
    "Molarity": (8, 16),
    "Silicate Solution": (100, 250),
    "Sand": (600, 900),
    "Coarse Aggregate": (800, 1100),
    "Water": (150, 250),
    "SP": (0.5, 3.5),
    "Temperature": (25, 85)
}

# Create feature name and bounds list
feature_names = list(feature_bounds.keys())
bounds = list(feature_bounds.values())

st.title("🧪 Inverse Design: SCGPC Mix Optimization")
st.markdown("Enter your target properties below to get an optimized mix design.")

# User input for target outputs
target_cs = st.number_input("🔧 Target Compressive Strength (MPa)", min_value=0.0, step=1.0)
target_sf = st.number_input("🔧 Target Slump Flow (mm)", min_value=0.0, step=1.0)
target_t500 = st.number_input("🔧 Target T500 Flow Time (s)", min_value=0.0, step=0.1)

if st.button("🔍 Generate Mix Design"):
    with st.spinner("Optimizing mix..."):

        target_real = np.array([target_cs, target_sf, target_t500])

        # Objective function to minimize
        def objective_function(x):
            x_real = np.array(x).reshape(1, -1)
            y_pred = model.predict(x_real)[0]

    # Scale the predictions to match target scale
            y_pred_scaled = [
                y_pred[0] / 100,    # CS
                y_pred[1] / 1000,   # SF
                y_pred[2] / 100     # T500
            ]

    # Use the globally defined target_scaled
            loss = np.linalg.norm(np.array(y_pred_scaled) - target_scaled.flatten())
            return loss



        # Differential Evolution optimization
        result = differential_evolution(objective_function, bounds)
        best_mix = result.x

        # Final prediction from optimized mix
        predicted = model.predict([best_mix])[0]  # Ensure input is 2D

        # Output dictionary
        mix_dict = {name: round(val, 2) for name, val in zip(feature_names, best_mix)}
        predicted_dict = {
            "C Strength": round(predicted[0], 2),
            "S flow": round(predicted[1], 2),
            "T 500": round(predicted[2], 2)
        }

        st.success("✅ Mix design successfully generated!")
        st.subheader("🎯 Predicted Properties (from optimized mix)")
        st.json(predicted_dict)

        st.subheader("🧱 Suggested Mix Design")
        st.table(mix_dict)
