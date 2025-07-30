import streamlit as st
import numpy as np
import joblib
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# Load model
model = joblib.load("best_model.pkl")

# Define feature bounds
feature_bounds = {
    "Fly Ash": (100, 600),
    "GGBS": (0, 300),
    "NaOH": (5, 100),
    "Molarity": (8, 16),
    "Silicate Solution": (50, 400),
    "Sand": (600, 1000),
    "Coarse Agg": (700, 1200),
    "Water": (100, 250),
    "SP": (0, 5),
    "Temperature": (25, 90)
}

feature_names = list(feature_bounds.keys())
bounds = list(feature_bounds.values())

st.title("🧱 Inverse Design: SCGPC Mix Prediction")

st.markdown("Enter desired target properties of geopolymer concrete:")

# Target inputs
desired_cs = st.number_input("Target Compressive Strength (MPa)", 20.0, 100.0, 50.0)
desired_sf = st.number_input("Target Slump Flow (mm)", 300.0, 800.0, 450.0)
desired_t500 = st.number_input("Target T500 Time (s)", 2.0, 30.0, 6.0)

# Scale targets
target_scaled = np.array([
    desired_cs / 100,       # CS scaled
    desired_sf / 1000,      # SF scaled
    desired_t500 / 100      # T500 scaled
]).reshape(1, -1)

# Objective function with T500 penalty
def objective_function(x):
    x_real = np.array(x).reshape(1, -1)
    y_pred = model.predict(x_real)[0]

    # Scale to real-world units
    y_pred_scaled = [
        y_pred[0] * 100,
        y_pred[1] * 1000,
        y_pred[2] * 100
    ]

    # Compute loss + penalty for high T500
    loss = np.linalg.norm(np.array(y_pred_scaled) - [desired_cs, desired_sf, desired_t500])
    
    if y_pred_scaled[2] > 10:
        penalty = (y_pred_scaled[2] - 10) ** 2
        loss += penalty

    return loss

# Run optimization
if st.button("🔍 Suggest Mix Design"):
    result = differential_evolution(objective_function, bounds, seed=42)
    best_mix = result.x

    mix_dict = {name: round(val, 2) for name, val in zip(feature_names, best_mix)}

    st.subheader("📋 Suggested Mix Design Proportions:")
    st.json(mix_dict)

    # Predict target output for best mix
    pred_scaled = model.predict(np.array(best_mix).reshape(1, -1))[0]
    predicted_output = {
        "C Strength": round(pred_scaled[0] * 100, 2),
        "S flow": round(pred_scaled[1] * 1000, 2),
        "T 500": round(pred_scaled[2] * 100, 2)
    }

    st.subheader("🎯 Predicted Performance:")
    st.write(predicted_output)

    # Donut chart visualization
    fig, ax = plt.subplots()
    labels = list(predicted_output.keys())
    values = list(predicted_output.values())

    colors = ['#4CAF50', '#2196F3', '#FF9800']
    wedges, texts = ax.pie(values, labels=labels, startangle=90, colors=colors, wedgeprops=dict(width=0.3))

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    ax.axis('equal')
    st.pyplot(fig)
