import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load your trained model
model = joblib.load("best_model.pkl")

st.title("Inverse Design of Self-Compacting Geopolymer Concrete Mix")

st.write("""
Input your target performance values below, and the app will predict suitable mix proportions.
""")

# User inputs for target properties (CS28, SF, T500)
cs_target = st.number_input("Compressive Strength (CS28) [MPa]", min_value=0.0, max_value=200.0, value=40.0)
sf_target = st.number_input("Slump Flow (SF) [mm]", min_value=0.0, max_value=1000.0, value=650.0)
t500_target = st.number_input("Flow Time (T500) [sec]", min_value=0.0, max_value=100.0, value=10.0)

if st.button("Predict Mix Design"):
    # Prepare input feature array
    X_input = np.array([[cs_target, sf_target, t500_target]])
    
    # Predict mix proportions
    predicted_mix = model.predict(X_input)
    
    mix_components = [
        "Fly Ash",
        "GGBS",
        "NaOH",
        "Molarity",
        "Silicate Solution",
        "Sand",
        "Coarse Aggregate",
        "Water",
        "Superplasticizer",
        "Temperature"
    ]
    
    # Convert predictions to DataFrame
    df_results = pd.DataFrame(predicted_mix, columns=mix_components)
    df_results = df_results.round(2)
    
    st.write("### Suggested Mix Proportions (kg or units):")
    st.dataframe(df_results.T.rename(columns={0:"Value"}))
    
    # Plot donut chart
    fig = go.Figure(data=[go.Pie(
        labels=mix_components,
        values=predicted_mix.flatten(),
        hole=0.5,
        hoverinfo="label+percent+value",
        textinfo="label+percent"
    )])
    fig.update_layout(title_text="Mix Composition Breakdown")
    st.plotly_chart(fig, use_container_width=True)
