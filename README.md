```markdown
# Ultra High Performance Concrete Strength Prediction Tool

## Overview
This application is a user-friendly GUI tool for predicting the compressive strength of Ultra High Performance Concrete (UHPC) and suggesting compositions based on target strength. It uses a trained machine learning model and allows users to save predictions and suggestions as PDF reports.

---

## Features
1. **Concrete Strength Prediction:**
   - Input various parameters like cement, silica fume, superplasticizer content, etc.
   - Get predictions for concrete compressive strength (MPa).

2. **Composition Suggestion:**
   - Enter a target compressive strength.
   - Get interpolated concrete mix compositions to achieve the target strength.

3. **PDF Report Generation:**
   - Save prediction and suggestion results as detailed PDF reports.

4. **Intuitive GUI Design:**
   - Interactive layout with input fields, buttons, and suggestions displayed in a Treeview.

5. **Comprehensive Guide:**
   - Integrated "How to Use" and "About" sections for user assistance.

---

## Requirements
- Python 3.8+
- Libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `joblib`
  - `fpdf`
  - `tkinter`
  - `Pillow`

---

## Installation
1. Clone or download the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the trained model (`best_model.pkl`) and scaler (`scaler.pkl`) are in the `./output_02/models/` directory.
4. Place the dataset file (`UHPC_Data.csv`) in the project directory.

---

## Usage
1. **Launch the Application:**
   - Run the Python script:
     ```bash
     python app.py
     ```

2. **Predict Strength:**
   - Fill in the input parameters and click **"Predict Strength"**.

3. **Suggest Composition:**
   - Enter a target strength and click **"Suggest Composition"**.

4. **Save Results:**
   - Save the predictions or suggested compositions as PDF reports using the save buttons.

---

## File Structure
- `app.py`: Main application script.
- `output_02/models/`: Directory containing the trained model (`best_model.pkl`) and scaler (`scaler.pkl`).
- `UHPC_Data.csv`: Dataset file for compositions and target strength.

---

## How to Use
### Input Parameters
Each parameter represents a specific material property. Below is a brief description:
- **C**: Cement Content (kg/m³)
- **S**: Silica Fume Content (kg/m³)
- **SF**: Superplasticizer Content (kg/m³)
- **LP**: Lime Powder Content (kg/m³)
- **QP**: Quartz Powder Content (kg/m³)
- **FA**: Fly Ash Content (kg/m³)
- **NS**: Nano Silica Content (kg/m³)
- **W**: Water Content (kg/m³)
- **Sand**: Fine Aggregate Content (kg/m³)
- **Gravel**: Coarse Aggregate Content (kg/m³)
- **Fi**: Fineness Modulus of Aggregates (unitless)
- **SP**: Specific Gravity of Aggregates (unitless)
- **RH**: Relative Humidity (%)
- **T**: Temperature (°C)
- **Age**: Age of Concrete (days)

### Buttons and Functionalities
- **Predict Strength**: Predict compressive strength based on input values.
- **Suggest Composition**: Suggest concrete mix compositions to achieve a target strength.
- **Clear Inputs**: Reset input fields.
- **Save Prediction as PDF**: Save the predicted strength and input parameters.
- **Save Suggestions as PDF**: Save the suggested compositions and target strength.
  
![GUI_03](https://github.com/user-attachments/assets/5d856d16-9311-400f-9b16-d08d764b5f6f)

---


## Future Improvements
- Add advanced models for prediction.
- Enhance visualization for suggested compositions.
- Provide batch processing for multiple inputs.

---

## License
This project is licensed under the **Creative Commons Zero v1.0 Universal**. For details, see [CC0 License](https://creativecommons.org/publicdomain/zero/1.0/).

---

## Contact
For questions, feedback, or contributions, contact **Rod Will** at **[rhudwill@gmail.com]**.
```
