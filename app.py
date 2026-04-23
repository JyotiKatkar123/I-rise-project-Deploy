import streamlit as st
import pickle
import numpy as np

# Set page configuration
st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# Custom CSS for Centering and Styling
st.markdown("""
    <style>
    /* Center the main title and description */
    .reportview-container .main .block-container {
        max-width: 800px;
        padding-top: 2rem;
    }
    h1, h3, p {
        text-align: center;
    }
    /* Style the Predict button to be centered and attractive */
    .stButton > button {
        display: block;
        margin: 0 auto;
        width: 50%;
        background-color: #007bff;
        color: white;
        border-radius: 10px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Header
st.markdown("<h1>Diabetes Prediction using ML</h1>", unsafe_allow_html=True)
st.markdown("<h3>Enter the details below</h3>", unsafe_allow_html=True)
st.write("---")

# Centering the input grid
col_left, col_mid, col_right = st.columns([1, 4, 1])

with col_mid:
    # Row 1
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, step=1, value=0)
    with row1_col2:
        glucose = st.number_input("Glucose", min_value=0.0, value=120.0)
    with row1_col3:
        blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=70.0)

    # Row 2
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        skin_thickness = st.number_input("Skin Thickness", min_value=0.0, value=20.0)
    with row2_col2:
        insulin = st.number_input("Insulin", min_value=0.0, value=80.0)
    with row2_col3:
        bmi = st.number_input("BMI", min_value=0.0, value=25.0)

    # Row 3 (Two items centered)
    row3_spacer1, row3_col1, row3_col2, row3_spacer2 = st.columns([0.5, 2, 2, 0.5])
    with row3_col1:
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f", value=0.471)
    with row3_col2:
        age = st.number_input("Age", min_value=0, step=1, value=33)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Predict Button
    if st.button("Predict"):
        # Features array mapped to the model inputs
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                              insulin, bmi, dpf, age]])
        
        prediction = model.predict(features)
        
        st.markdown("---")
        if prediction[0] == 1:
            st.error("### Result: The person is Diabetic")
        else:
            st.success("### Result: The person is Not Diabetic")

st.markdown("---")
st.markdown("<p style='color: grey;'>Data Analysis Project</p>", unsafe_allow_html=True)
