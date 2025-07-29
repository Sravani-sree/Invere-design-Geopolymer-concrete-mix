import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Page config
st.set_page_config(page_title="Inverse Design of SCGPC", layout="centered")

st.title("🧪 Inverse Design of Self-Compacting Geopolymer Concrete (SCGPC)")
st.markdown("Enter the **desired concrete performance targets** below:")

# --- Input Section ---
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        cs28 = st.number_input("🧱 Compressive Strength (CS28) [MPa]", min_value=0.0, step=0.1)
    with col2:
        sf = st.number_input("🌊 Slump Flow (SF) [mm]", min_value=0.0, step=0.1)
    with col3:
        t500 = st.number_input("⏱️ T500 Flow Time [sec]", min_value=0.0, step=0.1)
    submitted = st.form_submit_button("🔍 Predict Mix Design")

# --- Prediction Section ---
if submitted:
    try:
        # Load the bundled inverse model
        model_bundle = joblib.load("inverse_model_bundle.pkl")
        model = model_bundle["model"]
        input_scaler = model_bundle["input_scaler"]
        output_scaler = model_bundle["output_scaler"]

        # Scale and predict
        user_input = np.array([[cs28, sf, t500]])
        scaled_input = input_scaler.transform(user_input)
        prediction_scaled = model.predict(scaled_input)
        prediction = output_scaler.inverse_transform(prediction_scaled)

        # Mix Component Labels
        components = [
            "Fly Ash", "GGBS", "NaOH", "Molarity", "Silicate Solution",
            "Sand", "Coarse Aggregate", "Water", "Superplasticizer", "Curing Temp"
        ]
        result_dict = {comp: round(val, 2) for comp, val in zip(components, prediction[0])}
        df_result = pd.DataFrame([result_dict])

        # Show Results
        st.subheader("✅ Suggested Mix Design Proportions:")
        st.table(df_result)

        # --- Download Buttons ---
        csv = df_result.to_csv(index=False).encode("utf-8")
        json = df_result.to_json(orient="records", indent=2)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download CSV", csv, "predicted_mix.csv", "text/csv")
        with col2:
            st.download_button("📄 Download JSON", json, "predicted_mix.json", "application/json")

        # --- Charts ---
        st.subheader("📊 Mix Proportions - Bar Chart")
        st.bar_chart(df_result.T)

        st.subheader("🥧 Mix Proportions - Pie Chart")
        fig, ax = plt.subplots()
        ax.pie(df_result.iloc[0], labels=components, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error: {e}")
