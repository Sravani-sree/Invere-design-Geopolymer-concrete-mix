import streamlit as st
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# Load your bundled model + scalers from one file
bundle = joblib.load("scgpc_model_bundle.pkl")
model = bundle["model"]
input_scaler = bundle["input_scaler"]
output_scaler = bundle["output_scaler"]

# Set Streamlit page configuration
st.set_page_config(page_title="SCGPC Inverse Design", layout="wide")
st.title("🧪 Inverse Design of SCGPC Concrete Mix")
st.markdown("Enter the desired **target properties** below:")

# 🎯 Target properties input
cs_target = st.number_input("Compressive Strength (MPa)", min_value=10.0, max_value=80.0, value=40.0)
sf_target = st.number_input("Slump Flow (mm)", min_value=400.0, max_value=800.0, value=700.0)
t500_target = st.number_input("T500 Time (sec)", min_value=1.0, max_value=20.0, value=5.0)

# Transform target values to scaled version
target_values = np.array([[cs_target, sf_target, t500_target]])
scaled_target = output_scaler.transform(target_values)

# Define bounds for optimization (same as training data)
bounds = [
    (100, 600),   # Fly Ash
    (50, 300),    # GGBS
    (5, 40),      # NaOH
    (8, 16),      # Molarity
    (50, 300),    # Silicate Solution
    (600, 900),   # Sand
    (800, 1200),  # Coarse Aggregate
    (120, 220),   # Water
    (0.1, 5.0),   # Superplasticizer
    (20, 80),     # Temperature
]

input_labels = ['Fly Ash', 'GGBS', 'NaOH', 'Molarity', 'Silicate Solution',
                'Sand', 'Coarse Agg', 'Water', 'SP', 'Temperature']

# Define fitness function
def fitness(x):
    x_scaled = input_scaler.transform([x])
    y_pred_scaled = model.predict(x_scaled)
    error = np.mean((y_pred_scaled - scaled_target) ** 2)
    return error

# Button to start optimization
if st.button("🔍 Optimize Mix Design"):
    result = differential_evolution(fitness, bounds, seed=42, strategy='best1bin', maxiter=1000)
    optimal_mix = result.x
    predicted_scaled = model.predict(input_scaler.transform([optimal_mix]))
    predicted_output = output_scaler.inverse_transform(predicted_scaled)[0]

    # Create DataFrame for mix proportions
    mix_df = pd.DataFrame({
        "Component": input_labels,
        "Proportion": np.round(optimal_mix, 2)
    })

    # 📊 Display suggested mix proportions
    st.subheader("📊 Suggested Mix Design Proportions")
    st.dataframe(mix_df.set_index("Component"))

    # 📈 Display predicted output vs target
    st.subheader("📈 Model-Predicted Properties for Optimized Mix")
    col1, col2, col3 = st.columns(3)
    col1.metric("C Strength (MPa)", f"{predicted_output[0]:.2f}", f"{predicted_output[0] - cs_target:+.2f}")
    col2.metric("S Flow (mm)", f"{predicted_output[1]:.2f}", f"{predicted_output[1] - sf_target:+.2f}")
    col3.metric("T500 (sec)", f"{predicted_output[2]:.2f}", f"{predicted_output[2] - t500_target:+.2f}")

    # 📉 Bar chart: Target vs Predicted
    st.subheader("📉 Target vs Predicted Properties")
    targets = [cs_target, sf_target, t500_target]
    predicted = predicted_output
    labels = ["Compressive Strength", "Slump Flow", "T500 Time"]
    x = np.arange(len(labels))
    width = 0.35

    fig1, ax1 = plt.subplots()
    ax1.bar(x - width/2, targets, width, label='Target', color='skyblue')
    ax1.bar(x + width/2, predicted, width, label='Predicted', color='lightgreen')
    ax1.set_ylabel("Value")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_title("Target vs Predicted")
    ax1.legend()
    for i in range(len(x)):
        ax1.text(i - width/2, targets[i]+1, f"{targets[i]:.1f}", ha='center', fontsize=8)
        ax1.text(i + width/2, predicted[i]+1, f"{predicted[i]:.1f}", ha='center', fontsize=8)
    st.pyplot(fig1)

    # 🧯 Pie chart of mix proportions
    st.subheader("🧯 Mix Proportion Breakdown (Pie Chart)")
    fig2, ax2 = plt.subplots()
    ax2.pie(mix_df["Proportion"], labels=mix_df["Component"], autopct='%1.1f%%', startangle=140)
    ax2.axis("equal")
    st.pyplot(fig2)
