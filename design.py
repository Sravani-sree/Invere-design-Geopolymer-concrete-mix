# app.py
import streamlit as st
import numpy as np
import pickle
from scipy.optimize import minimize

# ------------------------------
# Load trained model
# ------------------------------
with open('xgb_inverse_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ------------------------------
# Define input columns
# ------------------------------
input_columns = [
    'Fly Ash',
    'GGBS',
    'NaOH',
    'Molarity',
    'Silicate Solution',
    'Sand',
    'Coarse Aggregate',
    'Water',
    'Spz',
    'Temperature'
]

# ------------------------------
# Define realistic bounds
# ------------------------------
bounds = [
    (300, 600),     # Fly Ash
    (50, 300),      # GGBS
    (5, 50),        # NaOH
    (6, 16),        # Molarity
    (50, 250),      # Silicate Solution
    (600, 900),     # Sand
    (700, 1200),    # Coarse Aggregate
    (140, 220),     # Water
    (0, 10),        # Spz
    (20, 80)        # Temperature
]

# ------------------------------
# Define objective function
# ------------------------------
def objective_function(inputs, target_values, model):
    inputs = np.array(inputs).reshape(1, -1)
    prediction = model.predict(inputs)[0]
    error = np.sum((prediction - target_values) ** 2)
    return error

# ------------------------------
# Inverse Design Optimizer
# ------------------------------
def inverse_design(target_values, model, bounds):
    initial_guess = np.array([(low + high) / 2 for low, high in bounds])
    result = minimize(
        objective_function,
        initial_guess,
        args=(target_values, model),
        bounds=bounds,
        method='L-BFGS-B'
    )
    if result.success:
        return result.x
    else:
        return None

# ------------------------------
# Streamlit UI
# ------------------------------
st.title('🧪 Geopolymer Concrete Inverse Design App')
st.markdown("""
Enter your **desired target properties** (compressive strength, slump flow, T500) 
and get **recommended mix design proportions**!
""")

# User inputs
cs_target = st.number_input('Desired Compressive Strength (MPa)', 20.0, 60.0, 35.0)
sf_target = st.number_input('Desired Slump Flow (mm)', 600.0, 800.0, 700.0)
t500_target = st.number_input('Desired T500 Flow Time (s)', 2.0, 5.0, 3.5)

if st.button('Run Inverse Design'):
    target_array = np.array([cs_target, sf_target, t500_target])
    optimal_mix = inverse_design(target_array, model, bounds)
    
    if optimal_mix is not None:
        st.success('✅ Optimized Mix Design Found!')
        mix_df = {col: [val] for col, val in zip(input_columns, optimal_mix)}
        st.table(mix_df)

        # Predict properties for this mix
        predicted_properties = model.predict(optimal_mix.reshape(1, -1))[0]
        st.subheader('Predicted Properties for Suggested Mix')
        st.write(f"- Compressive Strength: {predicted_properties[0]:.2f} MPa")
        st.write(f"- Slump Flow: {predicted_properties[1]:.2f} mm")
        st.write(f"- T500 Flow Time: {predicted_properties[2]:.2f} s")
    else:
        st.error('❌ Optimization failed. Try adjusting targets or check model.')

import pandas as pd

# Prepare DataFrame nicely
result_df = pd.DataFrame({
    'Component': input_columns,
    'Amount': [f"{val:.2f} kg/m³" if "Molarity" not in name and "Temperature" not in name else
               (f"{val:.2f} mol/L" if "Molarity" in name else f"{val:.2f} °C")
               for name, val in zip(input_columns, optimal_mix)]
})

import matplotlib.pyplot as plt

st.subheader("🥧 **Mix Design Pie Chart**")

# Prepare data for pie chart
labels = [name for name in input_columns if name != "Temperature"]
values = [val for name, val in zip(input_columns, optimal_mix) if name != "Temperature"]

# Define custom pastel colors
custom_colors = [
    '#FF9999','#66B3FF','#99FF99','#FFCC99',
    '#C2C2F0','#FFB266','#FF6666','#99CC99','#66CCCC'
]

# Create pie chart
fig, ax = plt.subplots(figsize=(7, 7))

wedges, texts, autotexts = ax.pie(
    values,
    labels=None,  # we'll use legend
    autopct='%1.1f%%',
    startangle=140,
    colors=custom_colors,
    pctdistance=0.85
)

# Draw center circle for "donut" look
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig.gca().add_artist(centre_circle)

# Add legend outside
ax.legend(
    wedges,
    labels,
    title="Mix Components",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1)
)

ax.set_title("Proportions of Mix Components (excluding Temperature)", fontsize=14)
plt.tight_layout()

# Display in Streamlit
st.pyplot(fig)


st.subheader("🔧 **Optimized Mix Design Proportions**")
st.dataframe(result_df.style.format({"Amount": str}))


