import streamlit as st
import numpy as np
import joblib
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# Load model
model = joblib.load("best_model.pkl")

# Define bounds for each input feature (example)
feature_bounds = {
    "Fly Ash": (300, 500),
    "GGBS": (50, 250),
    "NaOH": (10, 25),
    "Molarity": (8, 16),
    "Silicate Solution": (150, 250),
    "Sand": (600, 850),
    "Coarse Agg": (800, 1100),
    "Water": (150, 220),
    "SP": (1, 4),
    "Temperature": (25, 85)
}

feature_names = list(feature_bounds.keys())
bounds = list(feature_bounds.values())

st.title("🧪 Inverse Design of SCGPC Mix using ML + DE")

# Target Inputs
target_cs = st.number_input("Enter desired Compressive Strength (MPa)", 10.0, 80.0, 50.0)
target_sf = st.number_input("Enter desired Slump Flow (mm)", 300.0, 800.0, 450.0)
target_t500 = st.number_input("Enter desired T500 (sec)", 0.5, 10.0, 3.0)

# Scale target
target_scaled = np.array([
    target_cs / 100,
    target_sf / 1000,
    target_t500 / 100
]).reshape(1, -1)

# Objective Function
def objective_function(x):
    x_real = np.array(x).reshape(1, -1)
    y_pred = model.predict(x_real)[0]  # Already scaled output
    loss = np.linalg.norm(y_pred - target_scaled.flatten())
    return loss

# Run Optimization
if st.button("🔍 Optimize Mix Design"):
    with st.spinner("Optimizing... Please wait."):
        result = differential_evolution(objective_function, bounds, seed=42)
        best_mix = result.x
        y_pred_scaled = model.predict(best_mix.reshape(1, -1))[0]

        # Rescale predictions
        predicted_outputs = {
            "C Strength": round(y_pred_scaled[0] * 100, 2),
            "S flow": round(y_pred_scaled[1] * 1000, 2),
            "T 500": round(y_pred_scaled[2] * 100, 2)
        }

        # Display Results
        st.subheader("📋 Suggested Mix Design Proportions:")
        mix_dict = {name: round(val, 2) for name, val in zip(feature_names, best_mix)}
        st.json(mix_dict)

        st.subheader("📈 Predicted Target Properties:")
        st.write(predicted_outputs)

        # Donut Chart
        fig, ax = plt.subplots()
        labels = list(predicted_outputs.keys())
        sizes = list(predicted_outputs.values())
        colors = ['#4caf50', '#2196f3', '#ff9800']
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops=dict(width=0.4)
        )
        ax.axis('equal')
        plt.setp(autotexts, size=12, weight="bold")
        st.pyplot(fig)
