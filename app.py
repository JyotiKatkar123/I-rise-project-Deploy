import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(page_title="Diabetes Health Suite", layout="wide")

# Creative & Professional CSS
st.markdown("""
    <style>
    /* Main background and font styling */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Center the container */
    .main-container {
        max-width: 900px;
        margin: auto;
        padding: 2rem;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }

    /* Professional header */
    .main-title {
        color: #1E3A8A;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .sub-title {
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Style for the straight-line prediction */
    .result-text {
        text-align: center;
        font-size: 24px;
        font-weight: 600;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }

    /* Predict Button Styling */
    div.stButton > button:first-child {
        background-color: #1E3A8A;
        color: white;
        border-radius: 8px;
        width: 100%;
        height: 50px;
        font-size: 18px;
        border: none;
        transition: 0.3s;
    }
    
    div.stButton > button:hover {
        background-color: #3B82F6;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Header Section
st.markdown("<h1 class='main-title'>Diabetes Prediction AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Advanced Machine Learning Diagnostic Tool</p>", unsafe_allow_html=True)

# Form Container
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # 3-Column Grid for Inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0)
        dpf = st.number_input("Pedigree Function", min_value=0.0, format="%.3f")

    with col2:
        glucose = st.number_input("Glucose Level", min_value=0.0)
        insulin = st.number_input("Insulin Level", min_value=0.0)
        age = st.number_input("Age (Years)", min_value=0, step=1)

    with col3:
        blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
        bmi = st.number_input("BMI Value", min_value=0.0)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Prediction logic
    if st.button("Analyze Health Metrics"):
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                              insulin, bmi, dpf, age]])
        
        prediction = model.predict(features)
        
        # Straight line result display
        if prediction[0] == 1:
            st.markdown(f'<div class="result-text" style="background-color: #FEE2E2; color: #991B1B;">⚠️ Diagnostic Result: The patient is likely Diabetic.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-text" style="background-color: #DCFCE7; color: #166534;">✅ Diagnostic Result: The patient is likely Not Diabetic.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><p style='text-align: center; color: #9CA3AF; font-size: 12px;'>Confidential Data Processing | ML Model v1.0</p>", unsafe_allow_html=True)
