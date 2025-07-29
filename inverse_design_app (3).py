import streamlit as st
import numpy as np
import joblib
from scipy.optimize import differential_evolution

# Load the model and scalers from a bundled .pkl file
bundle = joblib.load("scgpc_model_bundle.pkl")
model = bundle["model"]
input_scaler = bundle["input_scaler"]
output_scaler = bundle["output_scaler"]

# Define realistic bounds for input features (based on your dataset or domain knowledge)
input_bounds = [
    (300, 600),   # Fly Ash
    (50, 300),    # GGBS
    (5, 40),      # NaOH
    (8, 16),      # Molarity
    (50, 300),    # Silicate Solution
    (600, 900),   # Sand
    (800, 1100),  # Coarse Aggregate
    (100, 250),   # Water
    (1, 6),       # Superplasticizer
    (25, 90),     # Temperature
]

# Function to make predictions
def predict_output(input_features):
    scaled_input = input_scaler.transform([input_features])
    scaled_output = model.predict(scaled_input)
    output = output_scaler.inverse_transform(scaled_output)
    return output[0]

# Inverse design objective: minimize error between predicted and desired output
def inverse_design(target_output):
    # Clip unrealistic user inputs to stay within known performance limits
    target_output = np.clip(target_output, [20, 400, 1], [80, 800, 10])
    scaled_target = output_scaler.transform([target_output])[0]

    def objective(x):
        scaled_x = input_scaler.transform([x])
        pred_scaled_y = model.predict(scaled_x)[0]
        return np.mean((pred_scaled_y - scaled_target) ** 2)

    result = differential_evolution(objective, input_bounds, strategy='best1bin', maxiter=1000, popsize=15)
    return result.x

# Streamlit UI
st.title("Inverse Design of Self-Compacting Geopolymer Concrete Mix")

st.sidebar.header("Target Performance Inputs")

cs = st.sidebar.number_input("Compressive Strength (CS28) [MPa]", min_value=20.0, max_value=80.0, value=50.0, step=0.5)
sf = st.sidebar.number_input("Slump Flow (SF) [mm]", min_value=400.0, max_value=800.0, value=650.0, step=5.0)
t500 = st.sidebar.number_input("T500 Flow Time [sec]", min_value=1.0, max_value=10.0, value=2.5, step=0.1)

if st.button("Suggest Mix Design"):
    target_output = [cs, sf, t500]
    mix = inverse_design(target_output)

    # Display results
    st.subheader("Suggested Mix Design Proportions")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Fly Ash: `{mix[0]:.2f}` kg")
        st.write(f"GGBS: `{mix[1]:.2f}` kg")
        st.write(f"NaOH: `{mix[2]:.2f}` kg")
        st.write(f"Molarity: `{mix[3]:.2f}` M")
        st.write(f"Silicate Solution: `{mix[4]:.2f}` kg")
    with col2:
        st.write(f"Sand: `{mix[5]:.2f}` kg")
        st.write(f"Coarse Aggregate: `{mix[6]:.2f}` kg")
        st.write(f"Water: `{mix[7]:.2f}` kg")
        st.write(f"Superplasticizer: `{mix[8]:.2f}` kg")
        st.write(f"Temperature: `{mix[9]:.2f}` °C")

    predicted = predict_output(mix)
    st.subheader("Predicted Properties for Suggested Mix")
    st.write(f"Compressive Strength (CS28): `{predicted[0]:.2f}` MPa")
    st.write(f"Slump Flow (SF): `{predicted[1]:.2f}` mm")
    st.write(f"T500 Flow Time: `{predicted[2]:.2f}` sec")
