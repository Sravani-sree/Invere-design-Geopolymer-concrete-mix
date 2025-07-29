import streamlit as st
from inverse_design_model import inverse_design

st.set_page_config(page_title="Inverse Design - SCGPC", layout="centered")

st.title("🧪 Inverse Design of Self-Compacting Geopolymer Concrete")
st.markdown("Enter target concrete properties to generate suitable mix design.")

# User input for target performance values
cs28 = st.number_input("Target Compressive Strength (MPa)", min_value=10.0, max_value=100.0, value=45.0)
sf = st.number_input("Target Slump Flow (mm)", min_value=400.0, max_value=800.0, value=650.0)
t500 = st.number_input("Target T500 Flow Time (s)", min_value=0.5, max_value=10.0, value=3.5)

if st.button("🔍 Generate Mix Design"):
    with st.spinner("Running inverse optimization..."):
        target = [cs28, sf, t500]
        mix_design, predicted_output = inverse_design(target)

    st.success("Mix design successfully generated!")
    
    st.subheader("🎯 Predicted Properties (from optimized mix)")
    st.json(predicted_output)

    st.subheader("🧱 Suggested Mix Design")
    for key, val in mix_design.items():
        st.write(f"**{key}**: {val:.2f}")
