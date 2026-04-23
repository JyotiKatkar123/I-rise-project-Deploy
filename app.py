import streamlit as st
import pickle
import numpy as np

# Set page configuration for a centered layout
st.set_page_config(page_title="Health Predictor", layout="wide")

# Custom CSS for centering and styling
st.markdown("""
    <style>
    .main {
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    div.row-widget.stRadio > div{
        flex-direction:row;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Header Section (Centered)
st.markdown("<h1 style='text-align: center;'>Health Diagnostic Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Please provide the following information for an instant analysis.</p>", unsafe_allow_html=True)
st.markdown("---")

# Centering the input form using columns
empty_col, main_col, empty_col2 = st.columns([1, 2, 1])

with main_col:
    # Categorical/Discrete values displayed as Select Boxes or Sliders
    pregnancies = st.selectbox("Number of Pregnancies", options=list(range(0, 21)))
    
    age = st.select_slider("Select Age", options=list(range(1, 101)), value=25)

    # Numerical inputs for health metrics
    col_a, col_b = st.columns(2)
    with col_a:
        glucose = st.number_input("Glucose Level", min_value=0.0, value=100.0)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, value=70.0)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, value=20.0)

    with col_b:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, value=80.0)
        bmi = st.number_input("BMI (Weight/Height²)", min_value=0.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f", value=0.500)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Predict Button
    if st.button("Run Diagnostic Analysis"):
        # Features must be in the exact order: 
        # [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                              insulin, bmi, dpf, age]])
        
        prediction = model.predict(features)
        
        st.markdown("---")
        if prediction[0] == 1:
            st.warning("### Result: High Risk Detected")
            st.info("The model suggests a high probability. Please consult a medical professional.")
        else:
            st.success("### Result: Low Risk Detected")
            st.info("The model suggests a low probability. Maintain your healthy lifestyle!")

# Footer
st.markdown("<br><p style='text-align: center; color: grey;'>Note: This is a machine learning demo and not a substitute for medical advice.</p>", unsafe_allow_html=True)
