import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="SCGPC Inverse Design", layout="centered")

st.title("🧪 Inverse Design of Self-Compacting Geopolymer Concrete (SCGPC)")
st.markdown("Enter your target concrete performance parameters below:")

# Input form for performance targets
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        cs28 = st.number_input("🧱 Compressive Strength (CS28) [MPa]", min_value=0.0, step=0.1)
    with col2:
        sf = st.number_input("🌊 Slump Flow (SF) [mm]", min_value=0.0, step=0.1)
    with col3:
        t500 = st.number_input("⏱️ T500 Flow Time [sec]", min_value=0.0, step=0.1)
    submitted = st.form_submit_button("🔍 Predict Mix Design")

if submitted:
    try:
        # Load the inverse model and scalers (combined file)
        model_bundle = joblib.load("inverse_model_bundle.pkl")
        model = model_bundle["model"]
        input_scaler = model_bundle["input_scaler"]
        output_scaler = model_bundle["output_scaler"]

        # Prepare input
        user_input = np.array([[cs28, sf, t500]])
        scaled_input = input_scaler.transform(user_input)

        # Predict
        prediction_scaled = model.predict(scaled_input)
        prediction = output_scaler.inverse_transform(prediction_scaled)

        # Define mix component labels
        components = [
            "Fly Ash", "GGBS", "NaOH", "Molarity", "Silicate Solution",
            "Sand", "Coarse Aggregate", "Water", "Superplasticizer", "Curing Temp"
        ]

        st.subheader("🔧 Suggested Mix Design Proportions:")
        result_dict = {comp: round(val, 2) for comp, val in zip(components, prediction[0])}
        st.table(result_dict)

    except Exception as e:
        st.error(f"⚠️ Something went wrong: {e}")
