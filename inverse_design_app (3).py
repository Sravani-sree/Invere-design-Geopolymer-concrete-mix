import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="SCGPC Inverse Design", layout="centered")

# Title
st.title("🔁 Inverse Design of SCGPC Concrete Mix")
st.markdown("Predict raw material mix proportions from desired concrete properties.")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("good_model.pkl")

model = load_model()

# Input sliders
cs = st.number_input("Compressive Strength (CS28, MPa)", min_value=10.0, max_value=100.0, step=1.0, value=40.0)
sf = st.number_input("Slump Flow (SF, mm)", min_value=500.0, max_value=850.0, step=1.0, value=650.0)
t500 = st.number_input("Flow Time (T500, sec)", min_value=1.0, max_value=10.0, step=0.1, value=3.0)

# Predict
if st.button("🔎 Predict Mix Design"):
    X_input = np.array([[cs, sf, t500]])
    try:
        prediction = model.predict(X_input)[0]
        mix_labels = ['Fly Ash', 'GGBS', 'NaOH', 'Molarity', 'Silicate Solution', 
                      'Sand', 'Coarse Agg', 'Water', 'SP', 'Temperature']
        
        result_df = pd.DataFrame([prediction], columns=mix_labels)
        st.success("✅ Prediction Successful!")

        # Show table
        st.subheader("📋 Predicted Mix Proportions")
        st.dataframe(result_df.T.rename(columns={0: "Amount"}), use_container_width=True)

        # Donut chart
        st.subheader("📊 Mix Proportion Distribution")
        fig, ax = plt.subplots()
        ax.pie(prediction, labels=mix_labels, startangle=90, counterclock=False, wedgeprops={'width': 0.4})
        ax.axis('equal')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
