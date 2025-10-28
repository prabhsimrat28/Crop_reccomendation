# === Smart Crop Recommendation Web App ===
import streamlit as st
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# === Load the pre-trained model ===
MODEL_PATH = 'RF.pkl'
if os.path.exists(MODEL_PATH):
    RF_Model = pickle.load(open(MODEL_PATH, 'rb'))
else:
    st.error("Model file 'RF.pkl' not found. Please upload it to the same directory as this script.")
    st.stop()

# === Prediction Function ===
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    """Predict the best crop using the trained Random Forest model"""
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    prediction = RF_Model.predict(input_data)
    return prediction[0]

# === Streamlit App UI ===
def main():
    st.markdown("<h1 style='text-align:center; color:green;'>🌾 Smart Crop Recommendation</h1>", unsafe_allow_html=True)
    st.sidebar.title("AgriSens – Input Crop Parameters")
    
    # Input fields
    nitrogen = st.sidebar.number_input("Nitrogen (N)", min_value=0.0, max_value=140.0, value=0.0, step=0.1)
    phosphorus = st.sidebar.number_input("Phosphorus (P)", min_value=0.0, max_value=145.0, value=0.0, step=0.1)
    potassium = st.sidebar.number_input("Potassium (K)", min_value=0.0, max_value=205.0, value=0.0, step=0.1)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=51.0, value=0.0, step=0.1)
    humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    ph = st.sidebar.number_input("Soil pH", min_value=0.0, max_value=14.0, value=0.0, step=0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)

    # Predict button
    if st.sidebar.button("Predict"):
        # Basic validation
        inputs = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        if np.isnan(inputs).any() or (inputs == 0).all():
            st.error("Please enter valid non-zero values for all inputs.")
        else:
            crop = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
            st.success(f"🌱 Recommended Crop: **{crop.capitalize()}**")

# === Run the app ===
if __name__ == '__main__':
    main()
