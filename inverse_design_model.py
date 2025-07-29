import numpy as np
import joblib
from scipy.optimize import differential_evolution

# Load the forward prediction model
model = joblib.load("best_model.pkl")

# Input bounds for each mix component
input_bounds = {
    'Fly Ash': (100, 500),
    'GGBS': (50, 300),
    'NaOH': (5, 50),
    'Molarity': (8, 16),
    'Silicate Solution': (50, 300),
    'Sand': (600, 900),
    'Coarse Agg': (800, 1100),
    'Water': (100, 250),
    'SP': (0, 5),
    'Temperature': (20, 90)
}
feature_names = list(input_bounds.keys())
bounds = list(input_bounds.values())

# Fitness function for optimization
def objective_function(inputs, target_values):
    inputs = np.array(inputs).reshape(1, -1)
    predicted = model.predict(inputs)[0]
    return np.mean((predicted - target_values) ** 2)

# Inverse design function
def inverse_design(target_values):
    result = differential_evolution(
        objective_function,
        bounds=bounds,
        args=(target_values,),
        strategy='best1bin',
        maxiter=300,
        popsize=15,
        seed=42,
        tol=1e-6,
        polish=True,
        disp=False
    )
    optimized_input = result.x
    predicted_output = model.predict([optimized_input])[0]

    mix_dict = dict(zip(feature_names, optimized_input))
    prediction_dict = dict(zip(["C Strength", "S flow", "T 500"], predicted_output))

    return mix_dict, prediction_dict
