import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import os

st.set_page_config(page_title="SCGPC Inverse Design", layout="wide")

st.title("🧪 Inverse Design of Self-Compacting Geopolymer Concrete Mix")

# Debug (optional)
# st.write("Current directory:", os.getcwd())
# st.write("Files in dir:", os.listdir())

# Load model
try:
    model = joblib.load("good_model.pkl")
except FileNotFoundError:
    st.error("❌ 'good_model.pkl' not found. Make sure it's uploaded with your app.")
    st.stop()

st.markdown("Input your **target properties**, and this app will predict a suitable mix design.")

# Input fields
col1, col2, col3 = st.columns(3)
with col1:
    cs = st.number_input("🧱 Compressive Strength (CS28) [MPa]", min_value=0.0, max_value=200.0, value=40.0)
with col2:
    sf = st.number_input("🌊 Slump Flow (SF) [mm]", min_value=0.0, max_value=1000.0, value=650.0)
with col3:
    t500 = st.number_input("⏱️ T500 Flow Time [s]", min_value=0.0, max_value=100.0, value=10.0)

if st.button("🔍 Predict Mix Design"):
    # Prepare input for prediction
    X_input = np.array([[cs, sf, t500]])

    try:
        prediction = model.predict(X_input)
    except ValueError as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Output labels
    mix_labels = [
        "Fly Ash", "GGBS", "NaOH", "Molarity", "Silicate Solution",
        "Sand", "Coarse Aggregate", "Water", "Superplasticizer", "Temperature"
    ]

    # Create DataFrame
    df_result = pd.DataFrame(prediction, columns=mix_labels).round(2)
    df_t = df_result.T.rename(columns={0: "Value"}).reset_index().rename(columns={"index": "Component"})

    st.subheader("📊 Predicted Mix Proportions")
    st.dataframe(df_t, use_container_width=True)

    # Donut chart
    fig = go.Figure(data=[go.Pie(
        labels=mix_labels,
        values=prediction.flatten(),
        hole=0.5,
        textinfo="label+percent",
        hoverinfo="label+value"
    )])
    fig.update_layout(title="🌀 Mix Composition Breakdown", margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
