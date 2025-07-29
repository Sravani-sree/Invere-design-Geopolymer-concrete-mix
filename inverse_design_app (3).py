import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.optimize import differential_evolution

# --------------------------
# Sample training data (use your actual data here)
# --------------------------
# Replace this with actual training data
X_train = pd.read_csv("X_train.csv")  # Provide correct path
y_train = pd.read_csv("y_train.csv")  # Provide correct path

# --------------------------
# Preprocess and scale
# --------------------------
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()
X_scaled = input_scaler.fit_transform(X_train)
y_scaled = output_scaler.fit_transform(y_train)

# --------------------------
# Train the model
# --------------------------
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
model.fit(X_scaled, y_scaled)

# --------------------------
# Feature names and bounds
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
# Streamlit GUI
# --------------------------
st.set_page_config(page_title="Inverse Design: SCGPC", layout="centered")
st.title("🧪 Inverse Design of Self-Compacting Geopolymer Concrete")

st.markdown("Provide your desired target properties:")

cs28 = st.number_input("Target Compressive Strength (MPa)", min_value=10.0, max_value=100.0, value=45.0)
sf = st.number_input("Target Slump Flow (mm)", min_value=400.0, max_value=800.0, value=650.0)
t500 = st.number_input("Target Flow Time T500 (sec)", min_value=1.0, max_value=10.0, value=2.5)

if st.button("Suggest Mix Design"):
    with st.spinner("Running optimization..."):
        optimized_mix, predicted_props = inverse_design([cs28, sf, t500], bounds)

    st.subheader("🧱 Suggested Mix Design Proportions")
    for name, value in zip(feature_names, optimized_mix):
        st.write(f"**{name}:** {value:.2f}")

    st.subheader("📈 Predicted Properties for Suggested Mix")
    st.write(f"**Compressive Strength (CS28):** {predicted_props[0]:.2f} MPa")
    st.write(f"**Slump Flow (SF):** {predicted_props[1]:.2f} mm")
    st.write(f"**Flow Time (T500):** {predicted_props[2]:.2f} sec")
