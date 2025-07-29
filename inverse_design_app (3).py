import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load the model
model = joblib.load("best_model.pkl")

# App title
st.title("Inverse Design of SCGPC Mix Using XGBoost")

st.markdown("### Enter Target Properties:")
cs = st.number_input("Compressive Strength (CS28) [MPa]", min_value=0.0, step=1.0, format="%.2f")
sf = st.number_input("Slump Flow (SF) [mm]", min_value=0.0, step=1.0, format="%.2f")
t500 = st.number_input("T500 Flow Time [s]", min_value=0.0, step=0.1, format="%.2f")

# Predict button
if st.button("Predict Mix Design"):
    try:
        input_features = np.array([[cs, sf, t500]])
        prediction = model.predict(input_features)[0]

        mix_labels = [
            "Fly Ash", "GGBS", "NaOH", "Molarity",
            "Silicate Solution", "Sand", "Coarse Aggregate",
            "Water", "Superplasticizer", "Temperature"
        ]
        df_result = pd.DataFrame([prediction], columns=mix_labels)

        st.success("Predicted Mix Proportions:")
        st.dataframe(df_result.style.format(precision=2), use_container_width=True)

        # Donut chart
        fig = go.Figure(data=[go.Pie(
            labels=mix_labels,
            values=prediction,
            hole=0.4,
            textinfo='label+percent'
        )])
        fig.update_layout(title="Mix Composition Breakdown")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
