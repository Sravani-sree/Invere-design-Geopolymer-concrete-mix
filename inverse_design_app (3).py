import pickle
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import differential_evolution

class GeopolymerMixDesigner:
    def __init__(self, root):
        self.root = root
        self.root.title("Geopolymer Mix Designer - Inverse Design System")
        
        # Load the pre-trained model
        try:
            with open('scgpc_model_bundle.pkl', 'rb') as f:
                self.model = pickle.load(f)
            self.feature_names = [
                'Fly Ash (kg/m³)', 'GGBS (kg/m³)', 'NaOH (kg/m³)', 
                'Molarity (M)', 'Silicate Solution (kg/m³)', 
                'Sand (kg/m³)', 'Coarse Aggregates (kg/m³)',
                'Water (kg/m³)', 'Superplasticizer (kg/m³)', 
                'Curing Temperature (°C)'
            ]
            self.target_names = [
                'Compressive Strength (MPa)', 
                'Slump Flow (mm)', 
                'T500 (s)'
            ]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.root.destroy()
            return
        
        # Set realistic parameter bounds (based on typical geopolymer concrete ranges)
        self.bounds = [
            (300, 500),    # Fly Ash (kg/m³)
            (0, 200),      # GGBS (kg/m³)
            (30, 80),      # NaOH (kg/m³)
            (8, 14),       # Molarity (M)
            (100, 200),    # Silicate Solution (kg/m³)
            (600, 800),    # Sand (kg/m³)
            (1000, 1200),  # Coarse Aggregates (kg/m³)
            (30, 70),      # Water (kg/m³)
            (0, 5),        # Superplasticizer (kg/m³)
            (25, 90)       # Curing Temperature (°C)
        ]
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Main frames
        self.input_frame = ttk.LabelFrame(self.root, text="Target Properties", padding=10)
        self.input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.constraint_frame = ttk.LabelFrame(self.root, text="Constraints", padding=10)
        self.constraint_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.result_frame = ttk.LabelFrame(self.root, text="Optimal Mix Design", padding=10)
        self.result_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        self.viz_frame = ttk.LabelFrame(self.root, text="Visualization", padding=10)
        self.viz_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # Target properties inputs
        ttk.Label(self.input_frame, text="Compressive Strength (MPa):").grid(row=0, column=0, sticky="w")
        self.cs_var = tk.DoubleVar(value=40.0)
        ttk.Entry(self.input_frame, textvariable=self.cs_var, width=10).grid(row=0, column=1)
        
        ttk.Label(self.input_frame, text="Slump Flow (mm):").grid(row=1, column=0, sticky="w")
        self.sf_var = tk.DoubleVar(value=600.0)
        ttk.Entry(self.input_frame, textvariable=self.sf_var, width=10).grid(row=1, column=1)
        
        ttk.Label(self.input_frame, text="T500 (s):").grid(row=2, column=0, sticky="w")
        self.t500_var = tk.DoubleVar(value=3.0)
        ttk.Entry(self.input_frame, textvariable=self.t500_var, width=10).grid(row=2, column=1)
        
        # Constraints
        ttk.Label(self.constraint_frame, text="Binder Type:").grid(row=0, column=0, sticky="w")
        self.binder_type = tk.StringVar(value="Fly Ash + GGBS")
        ttk.Combobox(self.constraint_frame, textvariable=self.binder_type, 
                    values=["Fly Ash Only", "GGBS Only", "Fly Ash + GGBS"], width=15).grid(row=0, column=1)
        
        ttk.Label(self.constraint_frame, text="FA:GGBS Ratio:").grid(row=1, column=0, sticky="w")
        self.binder_ratio = tk.StringVar(value="70:30")
        ttk.Combobox(self.constraint_frame, textvariable=self.binder_ratio, 
                    values=["100:0", "70:30", "50:50", "30:70", "0:100"], width=8).grid(row=1, column=1)
        
        ttk.Label(self.constraint_frame, text="NaOH Molarity:").grid(row=2, column=0, sticky="w")
        self.molarity = tk.StringVar(value="10M")
        ttk.Combobox(self.constraint_frame, textvariable=self.molarity, 
                    values=["8M", "10M", "12M", "14M"], width=8).grid(row=2, column=1)
        
        # Results display
        self.result_text = tk.Text(self.result_frame, height=15, width=40, state="disabled")
        self.result_text.grid(row=0, column=0, sticky="nsew")
        
        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Calculate Optimal Mix", command=self.optimize_mix).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Reset", command=self.reset).pack(side="left", padx=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=2)
        
        self.result_frame.columnconfigure(0, weight=1)
        self.result_frame.rowconfigure(0, weight=1)
        
    def optimize_mix(self):
        """Find the optimal mix design that matches target properties"""
        try:
            # Get target values
            targets = np.array([
                self.cs_var.get(),
                self.sf_var.get(),
                self.t500_var.get()
            ])
            
            # Apply constraints
            constraints = self.get_constraints()
            
            # Run optimization
            result = differential_evolution(
                self.objective_function,
                bounds=self.bounds,
                args=(targets, constraints),
                maxiter=100,
                popsize=15,
                tol=0.01,
                polish=True
            )
            
            if not result.success:
                messagebox.showwarning("Warning", "Could not find optimal mix with current constraints")
                return
            
            # Get optimal mix
            optimal_mix = result.x
            
            # Display results
            self.display_results(optimal_mix, targets)
            
            # Visualize results
            self.visualize_results(optimal_mix, targets)
            
        except Exception as e:
            messagebox.showerror("Error", f"Optimization failed: {str(e)}")
    
    def get_constraints(self):
        """Process user constraints"""
        constraints = {}
        
        # Binder type constraints
        binder_type = self.binder_type.get()
        if binder_type == "Fly Ash Only":
            constraints['GGBS'] = 0
        elif binder_type == "GGBS Only":
            constraints['Fly Ash'] = 0
        
        # Binder ratio
        if binder_type == "Fly Ash + GGBS":
            fa, ggbs = map(int, self.binder_ratio.get().split(":"))
            total = fa + ggbs
            constraints['Fly Ash'] = fa / total
            constraints['GGBS'] = ggbs / total
        
        # Molarity constraint
        constraints['Molarity'] = float(self.molarity.get()[:-1])
        
        return constraints
    
    def objective_function(self, x, targets, constraints):
        """Objective function to minimize difference between targets and predictions"""
        # Apply constraints
        x = self.apply_constraints(x, constraints)
        
        # Make prediction
        prediction = self.model.predict(x.reshape(1, -1))[0]
        
        # Calculate error (weighted sum of squared errors)
        weights = np.array([1.0, 0.5, 0.5])  # Higher weight for compressive strength
        error = np.sum(weights * (prediction - targets) ** 2)
        
        return error
    
    def apply_constraints(self, x, constraints):
        """Apply constraints to the solution vector"""
        x = x.copy()
        
        # Binder ratio constraints
        if 'Fly Ash' in constraints and 'GGBS' in constraints:
            total_binder = x[0] + x[1]
            x[0] = constraints['Fly Ash'] * total_binder
            x[1] = constraints['GGBS'] * total_binder
        elif 'Fly Ash' in constraints:
            x[1] = 0  # GGBS = 0
        elif 'GGBS' in constraints:
            x[0] = 0  # Fly Ash = 0
            
        # Molarity constraint
        if 'Molarity' in constraints:
            x[3] = constraints['Molarity']
            
        return x
    
    def display_results(self, mix, targets):
        """Display the optimal mix design results"""
        # Make prediction with optimal mix
        prediction = self.model.predict(mix.reshape(1, -1))[0]
        
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        
        # Display mix proportions
        self.result_text.insert(tk.END, "Optimal Mix Proportions:\n")
        self.result_text.insert(tk.END, "------------------------\n")
        for i, param in enumerate(self.feature_names):
            self.result_text.insert(tk.END, f"{param}: {mix[i]:.1f}\n")
        
        # Display predicted properties
        self.result_text.insert(tk.END, "\nPredicted Properties:\n")
        self.result_text.insert(tk.END, "------------------------\n")
        for i, target in enumerate(self.target_names):
            self.result_text.insert(tk.END, f"{target}: {prediction[i]:.1f} (Target: {targets[i]})\n")
        
        self.result_text.config(state="disabled")
    
    def visualize_results(self, mix, targets):
        """Create visualizations of the results"""
        # Clear previous visualization
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart of composition
        composition = mix[:7]  # First 7 parameters are materials
        composition_labels = self.feature_names[:7]
        
        # Filter out zero values
        nonzero_mask = composition > 0
        composition = composition[nonzero_mask]
        composition_labels = [label for label, mask in zip(composition_labels, nonzero_mask) if mask]
        
        ax1.pie(composition, labels=composition_labels, autopct='%1.1f%%')
        ax1.set_title('Mix Composition (kg/m³)')
        
        # Bar chart of targets vs predicted
        predicted = self.model.predict(mix.reshape(1, -1))[0]
        x = np.arange(len(self.target_names))
        width = 0.35
        
        ax2.bar(x - width/2, targets, width, label='Target', color='tab:blue')
        ax2.bar(x + width/2, predicted, width, label='Predicted', color='tab:orange')
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.target_names)
        ax2.set_title('Target vs Predicted Properties')
        ax2.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def reset(self):
        """Reset all inputs to default values"""
        self.cs_var.set(40.0)
        self.sf_var.set(600.0)
        self.t500_var.set(3.0)
        self.binder_type.set("Fly Ash + GGBS")
        self.binder_ratio.set("70:30")
        self.molarity.set("10M")
        
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state="disabled")
        
        # Clear visualization
        for widget in self.viz_frame.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GeopolymerMixDesigner(root)
    root.mainloop()
