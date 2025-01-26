import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
from fpdf import FPDF
from sklearn.preprocessing import StandardScaler
import tkinter.filedialog
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image, ImageTk

# Load the model and scaler
best_model = joblib.load("./output_02/models/best_model.pkl")
scaler = joblib.load("./output_02/models/scaler.pkl")
data = pd.read_csv("UHPC_Data.csv")


# Define input features and target column
input_columns = ["C", "S", "SF", "LP", "QP", "FA", "NS", "W", "Sand", "Gravel", "Fi", "SP", "RH", "T", "Age"]
input_descriptions = {
    "C": ("Cement Content", "kg/m³"),
    "S": ("Silica Fume Content", "kg/m³"),
    "SF": ("Superplasticizer Content", "kg/m³"),
    "LP": ("Lime Powder Content", "kg/m³"),
    "QP": ("Quartz Powder Content", "kg/m³"),
    "FA": ("Fly Ash Content", "kg/m³"),
    "NS": ("Nano Silica Content", "kg/m³"),
    "W": ("Water Content", "kg/m³"),
    "Sand": ("Fine Aggregate (Sand) Content", "kg/m³"),
    "Gravel": ("Coarse Aggregate (Gravel) Content", "kg/m³"),
    "Fi": ("Fineness Modulus of Aggregates", "unitless"),
    "SP": ("Specific Gravity of Aggregates", "unitless"),
    "RH": ("Relative Humidity", "%"),
    "T": ("Temperature", "°C"),
    "Age": ("Age of Concrete", "days")
}
target_column = "CS"

# Function to predict compressive strength
def predict_strength():
    try:
        inputs = [float(entry.get()) for entry in entries]
        inputs_scaled =  scaler.fit_transform([inputs])
        prediction = best_model.predict(inputs_scaled)[0]
        messagebox.showinfo("Prediction Result", f"Predicted Concrete Strength: {prediction:.2f} MPa")
        global prediction_result
        prediction_result = prediction
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from tkinter import messagebox

# Load the dataset
data = pd.read_csv('UHPC_Data.csv')

# Specify input columns and target column (assuming 'strength' is the target)
input_columns = ["C", "S", "SF", "LP", "QP", "FA", "NS", "W", "Sand", "Gravel", "Fi", "SP", "RH", "T", "Age"]  # adjust based on your data
target_column = 'CS'

# Preprocessing the data: Split into input features and target
X = data[input_columns].values
y = data[target_column].values

# Scaling the features (assuming your AI model requires scaled inputs)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a model (using Linear Regression here for simplicity; use your best AI model)
best_model = LinearRegression()
best_model.fit(X_scaled, y)

def suggest_compositions():
    try:
        # Retrieve the target strength from the user input
        target_strength = float(target_strength_entry.get())

        # Ensure target strength is valid
        if target_strength <= 0:
            messagebox.showerror("Error", "Target Strength must be a positive number.")
            return

        # Function to calculate error between predicted and target strength
        def calculate_error(composition):
            composition_scaled = scaler.transform([composition])  # Use transform instead of fit_transform
            predicted_strength = best_model.predict(composition_scaled)[0]
            return abs(predicted_strength - target_strength)

        # Search for compositions in the dataset closest to the target strength
        closest_compositions = []
        errors = []

        for _, row in data.iterrows():
            composition = row[input_columns].values
            error = calculate_error(composition)
            errors.append(error)

        # Sort the dataset by the error (closest match first)
        sorted_indices = np.argsort(errors)

        # Get the top 2 closest compositions
        closest_compositions.append(data.iloc[sorted_indices[0]][input_columns].values)
        closest_compositions.append(data.iloc[sorted_indices[1]][input_columns].values)

        # Interpolate between the closest compositions
        interpolated_compositions = []
        for i in range(5):  # Number of interpolated compositions
            alpha = i / 4.0  # Interpolation factor: 0, 0.25, 0.5, 0.75, 1
            interpolated_composition = (1 - alpha) * closest_compositions[0] + alpha * closest_compositions[1]
            interpolated_compositions.append(interpolated_composition)

        # Display results in the Treeview
        for i in tree.get_children():
            tree.delete(i)

        # Insert interpolated compositions into the Treeview
        for feature_index, feature in enumerate(input_columns):
            values = [f"{interpolated_compositions[i][feature_index]:.2f}" for i in range(len(interpolated_compositions))]
            tree.insert("", "end", values=(feature, *values))

        global final_suggestions
        final_suggestions = interpolated_compositions

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")



# Function to clear inputs
def clear_inputs():
    for entry in entries:
        entry.delete(0, tk.END)

# Function to clear suggestions
def clear_suggestions():
    for i in tree.get_children():
        tree.delete(i)

# Function to save predictions as a PDF
def save_prediction_pdf():
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Ultra High Performance Concrete Strength Prediction", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Predicted Strength: {prediction_result:.2f} MPa", ln=True)
        pdf.ln(10)
        pdf.cell(200, 10, txt="Input Parameters:", ln=True)
        for col, entry in zip(input_columns, entries):
            pdf.cell(200, 10, txt=f"{col}: {entry.get()} ({input_descriptions[col][1]})", ln=True)
        
        # Allow user to select location and name for saving the PDF
        file_path = asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            pdf.output(file_path)
            messagebox.showinfo("Success", "Prediction report saved as PDF.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save PDF: {e}")

# Function to save suggestions as a PDF
def save_suggestion_pdf():
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Concrete Composition Suggestion Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Target Strength: {target_strength_entry.get()} MPa", ln=True)
        pdf.ln(10)
        pdf.cell(200, 10, txt="Suggested Compositions:", ln=True)
        for feature_index, feature in enumerate(input_columns):
            values = [f"{final_suggestions[i][feature_index]:.2f}" for i in range(len(final_suggestions))]
            pdf.cell(200, 10, txt=f"{feature} ({input_descriptions[feature][1]}): {', '.join(values)}", ln=True)

        # Allow user to select location and name for saving the PDF
        file_path = asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            pdf.output(file_path)
            messagebox.showinfo("Success", "Suggestion report saved as PDF.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save PDF: {e}")

# Create the GUI window
root = tk.Tk()
root.title("Ultra High Performance Concrete Strength Prediction Tool")
root.geometry("800x650")
root.configure(bg="lightgray")

# # Load the .ico file for the window icon
ico = Image.open('./Rod-Will_Logo.png')
photo = ImageTk.PhotoImage(ico)

# Set the window icon
root.wm_iconphoto(False, photo)

# Function to show the 'About' window
def show_about():
    messagebox.showinfo("About", "This tool predicts concrete compressive strength and suggests compositions based on target strength.\nDeveloped by Rod Will.")


# Function to display the "How to Use" guide
def show_how_to_use():
    guide = "How to Use:\n\n"
    for feature in input_columns:
        description, unit = input_descriptions[feature]
        guide += f"{feature} ({description}): {unit}\n"
    
    messagebox.showinfo("How to Use", "1. Enter the input parameters.\n2. Click 'Predict Strength' to get the predicted strength.\n3. Enter the target strength and click 'Suggest Composition' to get suggestions.\n4. Save the results as PDF.\n\n" + guide)

# Logo
try:
    logo_path = "./Presentation1.png"
    logo = Image.open(logo_path).resize((150, 475), Image.Resampling.LANCZOS)
    logo_img = ImageTk.PhotoImage(logo)
    tk.Label(root, image=logo_img, bg="#e8f5e9").pack(side="right", padx=10, pady=10)
except Exception:
    tk.Label(root, text="Logo not available.", fg="red", bg="#e8f5e9").pack(side="right", padx=10, pady=10)

# Footer
footer = tk.Label(root, text="Developed by Rod Will", font=("Arial", 10, "italic"), bg="#e3f2fd", fg="gray")
footer.pack(side="bottom", pady=5)

# Style to simulate a grid in the Treeview
style = ttk.Style()
style.configure("Treeview",
                highlightthickness=0,
                bd=0,
                font=('Arial', 10),
                rowheight=25)
style.configure("Treeview.Heading",
                font=('Arial', 10, 'bold'),
                anchor="center")
style.layout("Treeview",
             [('Treeview.treearea', {'sticky': 'nswe'})])  # Add grid to each column

# Top Section with About and How to Use buttons
# Top Section with About and How to Use buttons
top_frame = tk.Frame(root, bg="lightgray")
top_frame.pack(fill="x", padx=10, pady=5)
tk.Button(top_frame, text="About", command=show_about, bg="lightblue", fg="black").pack(side="left", padx=5)
tk.Button(top_frame, text="How to Use", command=show_how_to_use, bg="lightblue", fg="black").pack(side="left", padx=5)

tk.Label(top_frame, text="Ultra High Performance Concrete Strength Tool", font=("Arial", 14, "bold"), bg="lightgray").pack(side="left", padx=5)


# Left Section: Input Parameters
left_frame = tk.LabelFrame(root, text="Input Parameters for Prediction", bg="lightblue")
left_frame.pack(side="left", fill="y", padx=10, pady=5)

entries = []
for col in input_columns:
    row = tk.Frame(left_frame, bg="lightblue")
    row.pack(fill="x", padx=5, pady=2)
    tk.Label(row, text=col, width=15, anchor="w", bg="lightblue").pack(side="left")
    entry = tk.Entry(row)
    entry.pack(side="right", fill="x", expand=True)
    entries.append(entry)

tk.Button(left_frame, text="Predict Strength", command=predict_strength, bg="green", fg="white").pack(pady=5)
tk.Button(left_frame, text="Clear Inputs", command=clear_inputs, bg="red", fg="white").pack(pady=5)
tk.Button(left_frame, text="Save Prediction as PDF", command=save_prediction_pdf, bg="blue", fg="white").pack(pady=5)

# Right Section: Suggest Compositions
right_frame = tk.LabelFrame(root, text="Suggest Composition for Target Strength", bg="lightgreen", relief="solid", bd=2)
right_frame.pack(side="left", fill="both", expand=True, padx=10, pady=5)

tk.Label(right_frame, text="Target Strength (MPa)", bg="lightgreen").pack(anchor="w", padx=5, pady=2)
target_strength_entry = tk.Entry(right_frame)
target_strength_entry.pack(fill="x", padx=5, pady=2)

tk.Button(right_frame, text="Suggest Composition", command=suggest_compositions, bg="green", fg="white").pack(pady=5)
tk.Button(right_frame, text="Clear Suggestions", command=clear_suggestions, bg="red", fg="white").pack(pady=5)
tk.Button(right_frame, text="Save Suggestion as PDF", command=save_suggestion_pdf, bg="blue", fg="white").pack(pady=5)

# Updated Treeview with gridlines
tree = ttk.Treeview(right_frame, columns=("Feature", "CS1", "CS2", "CS3", "CS4", "CS5"), show="headings", style="Treeview")
tree.heading("Feature", text="Feature")
tree.heading("CS1", text="CS1")
tree.heading("CS2", text="CS2")
tree.heading("CS3", text="CS3")
tree.heading("CS4", text="CS4")
tree.heading("CS5", text="CS5")
tree.column("Feature", width=55, anchor="center")
tree.column("CS1", width=55, anchor="center")
tree.column("CS2", width=55, anchor="center")
tree.column("CS3", width=55, anchor="center")
tree.column("CS4", width=55, anchor="center")
tree.column("CS5", width=55, anchor="center")

# Pack Treeview
tree.pack(fill="both", expand=True)


root.mainloop()
