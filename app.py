import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the Saved Model and Scaler
# This gets the absolute path of the current file (app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full paths to the models
model_path = os.path.join(current_dir, 'saved_models', 'lr_model.pkl')
scaler_path = os.path.join(current_dir, 'saved_models', 'scaler.pkl')

# Load the Saved Model and Scaler using the full paths
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    st.error(f"Error: Model files not found at {model_path}")
    st.stop()

# 2. Define the Prediction Function
def predict_performance(hours, scores, activities, sleep, papers):
    # Convert 'Yes'/'No' to 1/0
    act_numeric = 1 if activities == "Yes" else 0
    
    # Create a dataframe for input
    # Note: We must ensure columns match the training order exactly
    input_data = pd.DataFrame({
        'Hours Studied': [hours],
        'Previous Scores': [scores],
        'Extracurricular Activities': [act_numeric], # Note: Check if your model expects 'Extracurricular Activities_Yes' or just the raw column if you processed it differently. 
                                                     # Based on your previous code, you used get_dummies which creates 'Extracurricular Activities_Yes'.
                                                     # Let's align with that:
        'Extracurricular Activities_Yes': [act_numeric],
        'Sleep Hours': [sleep],
        'Sample Question Papers Practiced': [papers]
    })
    
    # Ensure we strictly select only the columns the model expects
    # (This handles the dummy variable naming convention)
    expected_cols = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities_Yes', 'Sleep Hours', 'Sample Question Papers Practiced']
    input_data = input_data[expected_cols]

    # Scale the numerical columns using the loaded scaler
    cols_to_scale = ['Hours Studied', 'Previous Scores']
    input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])

    # Predict
    prediction = model.predict(input_data)[0]
    return prediction

def get_grade_classification(score):
    if score >= 80: return "Excellent"
    elif score >= 60: return "Good"
    elif score >= 40: return "Average"
    else: return "Poor"

# 3. Streamlit UI Layout
st.title("Student Performance Predictor")
st.write("Enter the student's details below to predict their performance index.")

# Create columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    hours = st.slider("Hours Studied per Day", 1, 10, 5)
    scores = st.slider("Previous Scores", 40, 100, 70)
    activities = st.radio("Extracurricular Activities", ["Yes", "No"])

with col2:
    sleep = st.slider("Sleep Hours", 0, 10, 7)
    papers = st.slider("Sample Papers Practiced", 0, 10, 5)

# Prediction Button
if st.button("Predict Performance"):
    result = predict_performance(hours, scores, activities, sleep, papers)
    classification = get_grade_classification(result)
    
    st.success(f"Predicted Index: {result:.2f}")
    st.info(f"Classification: {classification}")
    
    # Optional: Logic to show specific advice based on score
    if classification == "Excellent":
        st.write("🌟 Great job! Keep up the consistency.")
    elif classification == "Poor":
        st.error("⚠️ This student may need extra help and study time.")

# Footer info
st.markdown("---")
st.markdown("""
**Grading Guidelines:**
* **80-100:** Excellent
* **60-79:** Good
* **40-59:** Average
* **<40:** Poor
""")
