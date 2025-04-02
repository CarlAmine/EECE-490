import streamlit as st
import joblib
import numpy as np
import requests
import os

# --- Download the model file from Google Drive if not already present ---
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
model = model_data["model"]            # TensorFlow/Keras model
vectorizer = model_data["vectorizer"]  # TF-IDF Vectorizer
label_encoder = model_data["label_encoder"]  # Label Encoder

# --- Configure Page ---
st.set_page_config(page_title="NourishWise", page_icon="🍽️", layout="wide")

# --- Custom CSS ---
custom_css = """
<style>
/* Set background gradient */
body {
    background: linear-gradient(135deg, #FDEFF9, #FFF0F3);
}

/* Header styling */
h1 {
    color: #FF4B4B;
    text-align: center;
    font-family: 'Roboto', sans-serif;
    font-weight: 700;
}

/* General text styling */
body, .css-1d391kg, .css-1aumxhk {
    font-family: 'Roboto', sans-serif;
}

/* Sidebar styling */
.css-1d391kg {
    background-color: #fff;
    border-right: 1px solid #e6e6e6;
}

/* Button styling */
.stButton > button {
    background-color: #FF4B4B;
    color: white;
    font-size: 18px;
    padding: 10px 20px;
    border-radius: 8px;
    border: none;
    transition: background-color 0.3s ease;
}
.stButton > button:hover {
    background-color: #E63946;
}

/* Footer styling */
footer {
    text-align: center;
    padding: 10px;
    color: #666;
    font-size: 14px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- App Header ---
st.markdown("<h1>🍽️ NourishWise: AI-Powered Recipe Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #555;'>Discover delicious recipes based on your ingredients.</p>", unsafe_allow_html=True)

# --- Layout: Use Sidebar for Ingredients ---
with st.sidebar:
    st.header("Input Ingredients")
    ingredients = st.text_area("📝 Enter Ingredients (comma-separated):", height=150)
    st.markdown("---")
    st.info("Please enter your available ingredients to receive a recipe recommendation.")

# Main Content Area
col1, col2 = st.columns([3, 1])
with col1:
    st.image("recipe1.jpg", use_column_width=True)  # Ensure the image file is in your app directory

with col2:
    st.markdown("<h3 style='text-align: center; color: #FF4B4B;'>Featured Recipe</h3>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1543353071-873f17a7a088?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=60", use_column_width=True)
    
st.markdown("---")

# --- Prediction Logic ---
if st.button("🔮 Predict Recipe"):
    if not ingredients.strip():
        st.error("⚠️ Please enter some ingredients in the sidebar.")
    else:
        try:
            X_input = vectorizer.transform([ingredients]).toarray()
            y_pred = model.predict(X_input)
            recipe_index = np.argmax(y_pred, axis=1)
            predicted_recipe = label_encoder.inverse_transform(recipe_index)
            
            st.success(f"✅ Recommended Recipe: **{predicted_recipe[0]}**")
            st.balloons()  # Fun visual effect!
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# --- Custom Footer ---
st.markdown("""
    <footer>
        <p>&copy; 2025 NourishWise. All rights reserved.</p>
    </footer>
""", unsafe_allow_html=True)
