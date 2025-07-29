import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Set page config
st.set_page_config(page_title="SCGPC Inverse Design", layout="centered")

st.title("🧪 Inverse Design of Self-Compacting Geopolymer Concrete")
st.markdown("Provide your desired performance targets below to get recommended mix design.")

# Load model
model = joblib.load("good_model.pkl")

# Input form
with st.form("inverse_form"):
    cs = st.number_input("Compressive Strength (CS28) [MPa]", min_value=0.0, value=35.0)
    sf = st.number_input("Slump Flow (SF) [mm]", min_value=0.0, value=700.0)
    t500 = st.number_input("T500 Flow Time [sec]", min_value=0.0, value=4.0)
    submit = st.form_submit_button("Predict Mix Design")

# Predict and display
if submit:
    X_input = np.array([[cs, sf, t500]])
    predicted_mix = model.predict(X_input)[0]

    mix_features = [
        "Fly Ash", "GGBS", "NaOH", "Molarity", "Silicate Sol.",
        "Sand", "Coarse Agg.", "Water", "SP", "Curing Temp"
    ]

    result_df = pd.DataFrame([predicted_mix], columns=mix_features).T
    result_df.columns = ["Predicted Value"]
    result_df["Predicted Value"] = result_df["Predicted Value"].round(2)

    st.subheader("🔧 Suggested Mix Design Proportions:")
    st.table(result_df)
