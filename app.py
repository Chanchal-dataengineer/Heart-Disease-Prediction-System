import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

title = "Heart Disease Prediction App"
st.title('heart Disease Prediction System')
st.write("This app predicts the likelihood of heart disease based on user input.")

# user input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=1, max_value=300, value=120)
chol = st.number_input("Serum Cholesterol", min_value=1, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar", options=[0, 1], format_func=lambda x: "False" if x == 0 else "True")
restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3])

# prediction
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal,]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    
    if prediction[0]==1:
        st.warning("The model predict a higher likleihood of heart disease.This is not medical diagnosis. please consult a healthcare professional for proper evalutions.")
    else:
        st.success("The model predicts a lower likelihood of heart disease.This is not medical diagnosis")