import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.title("Inverse Design of Self-Compacting Geopolymer Concrete (SCGPC)")

st.markdown("""
This tool predicts **raw material proportions** based on desired concrete properties:
- **Compressive Strength** (C Strength)
- **Slump Flow** (S flow)
- **T500 Flow Time** (T 500)
""")

# Input form for target values
with st.form("inverse_form"):
    c_strength = st.number_input("Target Compressive Strength (MPa)", min_value=0.0, step=0.1)
    s_flow = st.number_input("Target Slump Flow (mm)", min_value=0.0, step=1.0)
    t_500 = st.number_input("Target T500 Flow Time (s)", min_value=0.0, step=0.1)
    submitted = st.form_submit_button("Predict Mix Design")

if submitted:
    input_df = pd.DataFrame([[c_strength, s_flow, t_500]], columns=["C Strength", "S flow", "T 500"])
    try:
        prediction = model.predict(input_df)[0]
        mix_columns = [
            "Fly Ash", "GGBS", "NaOH", "Molarity", "Silicate solution",
            "sand", "coarse aggregates", "water", "Spz", "Temperature"
        ]
        result_df = pd.DataFrame([prediction], columns=mix_columns)

        st.success("Predicted Mix Proportions:")
        st.dataframe(result_df.T.rename(columns={0: "Quantity"}))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
