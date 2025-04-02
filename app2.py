import streamlit as st
import joblib
import numpy as np
import requests

import os

def download_file_from_google_drive(file_id, destination):
    if not os.path.exists(destination):  # Only download if the file doesn't exist
        URL = f'https://drive.google.com/uc?id={file_id}'
        session = requests.Session()
        response = session.get(URL, stream=True)
        
        if 'Content-Disposition' in response.headers:
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(32768):
                    f.write(chunk)

file_id = '1tR6_8S-yISRKZR2QJ6BjU3A3V5HHUCm7'
download_file_from_google_drive(file_id, 'recipe_model.pkl')

# --- Load Model and Preprocessors from a Single File ---
model_data = joblib.load("recipe_model.pkl")
model = model_data["model"]  # TensorFlow/Keras model
vectorizer = model_data["vectorizer"]  # TF-IDF Vectorizer
label_encoder = model_data["label_encoder"]  # Label Encoder

# --- Streamlit App UI ---
st.set_page_config(page_title="NourishWise", page_icon="🍽️", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        h1 {
            color: #FF4B4B;
            text-align: center;
        }
        .stButton > button {
            background-color: #FF4B4B;
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 8px;
            border: none;
        }
        .stButton > button:hover {
            background-color: #E63946;
        }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown("<h1>🍽️ NourishWise: AI-Powered Recipe Predictor</h1>", unsafe_allow_html=True)
st.write("Enter ingredients, and our AI will recommend the best recipe for you!")

# --- User Input ---
col1, col2 = st.columns([3, 1])  # Layout adjustment
with col1:
    ingredients = st.text_area("📝 Enter Ingredients (comma-separated):", height=150)
with col2:
    st.image("recipe1.jpg", use_column_width=True)  # Add a relevant image

# --- Prediction Logic ---
if st.button("🔮 Predict Recipe"):
    if not ingredients.strip():
        st.error("⚠️ Please enter some ingredients.")
    else:
        try:
            X_input = vectorizer.transform([ingredients]).toarray()
            y_pred = model.predict(X_input)
            recipe_index = np.argmax(y_pred, axis=1)
            predicted_recipe = label_encoder.inverse_transform(recipe_index)

            st.success(f"✅ Recommended Recipe: **{predicted_recipe[0]}**")
            st.balloons()  # 🎈 Adds a cool animation effect
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
