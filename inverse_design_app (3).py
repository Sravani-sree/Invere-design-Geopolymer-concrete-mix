import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load the saved model
model = joblib.load("good_model.pkl")

st.title("Inverse Design of Geopolymer Concrete Mix")

# Input target properties
cs_target = st.number_input("Compressive Strength (CS28) [MPa]", 0.0, 200.0, 40.0)
sf_target = st.number_input("Slump Flow (SF) [mm]", 0.0, 1000.0, 650.0)
t500_target = st.number_input("Flow Time (T500) [sec]", 0.0, 100.0, 10.0)

if st.button("Predict Mix Design"):
    X_input = np.array([[cs_target, sf_target, t500_target]])
    predicted_mix = model.predict(X_input)
    
    mix_components = [
        "Fly Ash", "GGBS", "NaOH", "Molarity", "Silicate Solution",
        "Sand", "Coarse Aggregate", "Water", "Superplasticizer", "Temperature"
    ]
    
    df_results = pd.DataFrame(predicted_mix, columns=mix_components).round(2)
    
    st.write("### Suggested Mix Proportions (kg or units):")
    st.dataframe(df_results.T.rename(columns={0: "Value"}))
    
    fig = go.Figure(data=[go.Pie(
        labels=mix_components,
        values=predicted_mix.flatten(),
        hole=0.5,
        hoverinfo="label+percent+value",
        textinfo="label+percent"
    )])
    fig.update_layout(title_text="Mix Composition Breakdown")
    st.plotly_chart(fig)
