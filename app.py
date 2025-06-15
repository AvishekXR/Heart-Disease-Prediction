import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# Load model
with open('pipe.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")

# Custom Style
st.markdown("""
    <style>
        .main {
            background-color: #e3f2fd;
            color: #0d47a1;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stButton>button {
            background-color: #1976d2;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5em 1em;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1565c0;
            color: #e3f2fd;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #0d47a1;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Header Section with main image
image = Image.open("ht.jpg")
st.image(image, use_container_width=True)
st.title("🫀 Heart Disease Risk Predictor")
st.write("An AI-driven clinical tool to estimate heart disease risk based on patient vitals and symptoms.")

# Input Fields
st.header("📝 Enter Patient Details")

def user_input():
    Age = st.number_input("Age", min_value=1, max_value=120, value=45)
    Sex = st.selectbox("Sex", ["M", "F"], help="M: Male, F: Female")
    ChestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"],
                                  help="ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic, TA: Typical Angina")
    RestingBP = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=200, value=120)
    Cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
    FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], help="LVH: Left Ventricular Hypertrophy")
    MaxHR = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    ExerciseAngina = st.selectbox("Exercise Induced Angina", ["Y", "N"], help="Y: Yes, N: No")
    Oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, step=0.1, value=1.0)
    ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], help="Slope of the peak exercise ST segment")

    data = pd.DataFrame({
        'Age': [Age],
        'Sex': [Sex],
        'ChestPainType': [ChestPainType],
        'RestingBP': [RestingBP],
        'Cholesterol': [Cholesterol],
        'FastingBS': [FastingBS],
        'RestingECG': [RestingECG],
        'MaxHR': [MaxHR],
        'ExerciseAngina': [ExerciseAngina],
        'Oldpeak': [Oldpeak],
        'ST_Slope': [ST_Slope]
    })
    return data

input_df = user_input()

# Prediction
if st.button("🧪 Predict Heart Disease Risk"):
    probability = model.predict_proba(input_df)[0][1]  # Probability of heart disease
    prediction = model.predict(input_df)[0]

    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk: Heart Disease Likely\n\nProbability: {probability*100:.2f}%")
    else:
        st.success(f"✅ Low Risk: No Heart Disease Detected\n\nProbability: {probability*100:.2f}%")

    st.markdown("---")
    st.info("This prediction is based on clinical parameters. For diagnosis, please consult a licensed cardiologist.")

# Footer
st.markdown("---")
st.caption("© 2025 | Built by Avishek Guragain | Powered by Streamlit & Scikit-learn")
