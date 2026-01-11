import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. Dynamic Path Loading ---
# This ensures the app finds the models no matter where it is running
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'saved_models', 'lr_model.pkl')
scaler_path = os.path.join(current_dir, 'saved_models', 'scaler.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    st.error(f"Error: Model files not found. Check if {model_path} exists.")
    st.stop()

# --- 2. Prediction Function with Auto-Ordering ---
def predict_performance(hours, scores, activities, sleep, papers):
    # Convert 'Yes'/'No' to 1/0
    act_numeric = 1 if activities == "Yes" else 0
    
    # Create the DataFrame with all potential columns
    input_data = pd.DataFrame({
        'Hours Studied': [hours],
        'Previous Scores': [scores],
        'Extracurricular Activities_Yes': [act_numeric],
        'Sleep Hours': [sleep],
        'Sample Question Papers Practiced': [papers]
    })
    
    # --- THE FIX: Ask the model for the correct order ---
    try:
        # 'feature_names_in_' is saved inside the model during training
        expected_cols = model.feature_names_in_
        input_data = input_data[expected_cols]
    except AttributeError:
        st.error("Error: The model does not have feature names saved. Please re-train using scikit-learn v1.0+")
        st.stop()
    except KeyError as e:
        st.error(f"Error: Missing column {e}. The model expects: {expected_cols}")
        st.stop()

    # Scale the numerical columns
    # We use the loaded scaler to transform only the specific columns
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

# --- 3. Streamlit UI Layout ---
st.title("Student Performance Predictor")
st.write("Enter the student's details below to predict their performance index.")

col1, col2 = st.columns(2)

with col1:
    hours = st.slider("Hours Studied per Day", 1, 10, 5)
    scores = st.slider("Previous Scores", 40, 100, 70)
    activities = st.radio("Extracurricular Activities", ["Yes", "No"])

with col2:
    sleep = st.slider("Sleep Hours", 0, 10, 7)
    papers = st.slider("Sample Papers Practiced", 0, 10, 5)

if st.button("Predict Performance"):
    # Run the prediction
    result = predict_performance(hours, scores, activities, sleep, papers)
    classification = get_grade_classification(result)
    
    # Display Result
    st.success(f"Predicted Index: {result:.2f}")
    st.info(f"Classification: {classification}")
    
    if classification == "Excellent":
        st.balloons()
        st.write("🌟 Great job! Keep up the consistency.")
    elif classification == "Poor":
        st.error("⚠️ This student may need extra help and study time.")

st.markdown("---")
st.markdown("""
**Grading Guidelines:**
* **80-100:** Excellent
* **60-79:** Good
* **40-59:** Average
* **<40:** Poor
""")
