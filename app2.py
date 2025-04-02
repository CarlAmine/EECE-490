# import streamlit as st
# import joblib
# import numpy as np
# import requests
# import os

# # --- Download the model file from Google Drive if not already present ---
# def download_file_from_google_drive(file_id, destination):
#     if not os.path.exists(destination):  # Only download if the file doesn't exist
#         URL = f'https://drive.google.com/uc?id={file_id}'
#         session = requests.Session()
#         response = session.get(URL, stream=True)
        
#         if 'Content-Disposition' in response.headers:
#             with open(destination, 'wb') as f:
#                 for chunk in response.iter_content(32768):
#                     f.write(chunk)

# file_id = '1tR6_8S-yISRKZR2QJ6BjU3A3V5HHUCm7'
# download_file_from_google_drive(file_id, 'recipe_model.pkl')

# # --- Load Model and Preprocessors from a Single File ---
# model_data = joblib.load("recipe_model.pkl")
# model = model_data["model"]            # TensorFlow/Keras model
# vectorizer = model_data["vectorizer"]  # TF-IDF Vectorizer
# label_encoder = model_data["label_encoder"]  # Label Encoder

# # --- Configure Page ---
# st.set_page_config(page_title="NourishWise", page_icon="🍽️", layout="wide")

# # --- Custom CSS ---
# custom_css = """
# <style>
# /* Set background gradient */
# body {
#     background: linear-gradient(135deg, #FDEFF9, #FFF0F3);
# }

# /* Header styling */
# h1 {
#     color: #FF4B4B;
#     text-align: center;
#     font-family: 'Roboto', sans-serif;
#     font-weight: 700;
# }

# /* General text styling */
# body, .css-1d391kg, .css-1aumxhk {
#     font-family: 'Roboto', sans-serif;
# }

# /* Sidebar styling */
# .css-1d391kg {
#     background-color: #fff;
#     border-right: 1px solid #e6e6e6;
# }

# /* Button styling */
# .stButton > button {
#     background-color: #FF4B4B;
#     color: white;
#     font-size: 18px;
#     padding: 10px 20px;
#     border-radius: 8px;
#     border: none;
#     transition: background-color 0.3s ease;
# }
# .stButton > button:hover {
#     background-color: #E63946;
# }

# /* Footer styling */
# footer {
#     text-align: center;
#     padding: 10px;
#     color: #666;
#     font-size: 14px;
# }
# </style>
# """
# st.markdown(custom_css, unsafe_allow_html=True)

# # --- App Header ---
# st.markdown("<h1>🍽️ NourishWise: AI-Powered Recipe Predictor</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center; font-size: 18px; color: #555;'>Discover delicious recipes based on your ingredients.</p>", unsafe_allow_html=True)

# # --- Layout: Use Sidebar for Ingredients ---
# with st.sidebar:
#     st.header("Input Ingredients")
#     ingredients = st.text_area("📝 Enter Ingredients (comma-separated):", height=150)
#     st.markdown("---")
#     st.info("Please enter your available ingredients to receive a recipe recommendation.")

# # Main Content Area
# col1, col2 = st.columns([3, 1])
# with col1:
#     st.image("recipe1.jpg", use_container_width=True)  # Updated parameter

# with col2:
#     st.markdown("<h3 style='text-align: center; color: #FF4B4B;'>Featured Recipe</h3>", unsafe_allow_html=True)
#     st.image("https://images.unsplash.com/photo-1543353071-873f17a7a088?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=60", use_container_width=True)
    
# st.markdown("---")

# # --- Prediction Logic ---
# if st.button("🔮 Predict Recipe"):
#     if not ingredients.strip():
#         st.error("⚠️ Please enter some ingredients in the sidebar.")
#     else:
#         try:
#             X_input = vectorizer.transform([ingredients]).toarray()
#             y_pred = model.predict(X_input)
#             recipe_index = np.argmax(y_pred, axis=1)
#             predicted_recipe = label_encoder.inverse_transform(recipe_index)
            
#             st.success(f"✅ Recommended Recipe: **{predicted_recipe[0]}**")
#             st.balloons()  # Fun visual effect!
#         except Exception as e:
#             st.error(f"❌ Error: {str(e)}")

# # --- Custom Footer ---
# st.markdown("""
#     <footer>
#         <p>&copy; 2025 NourishWise. All rights reserved.</p>
#     </footer>
# """, unsafe_allow_html=True)

import streamlit as st
import joblib
import numpy as np
import requests
import os

# --- Model Loading (Same as Before) ---
def download_file_from_google_drive(file_id, destination):
    if not os.path.exists(destination):
        URL = f'https://drive.google.com/uc?id={file_id}'
        session = requests.Session()
        response = session.get(URL, stream=True)
        if 'Content-Disposition' in response.headers:
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(32768):
                    f.write(chunk)

file_id = '1tR6_8S-yISRKZR2QJ6BjU3A3V5HHUCm7'
download_file_from_google_drive(file_id, 'recipe_model.pkl')

model_data = joblib.load("recipe_model.pkl")
model = model_data["model"]
vectorizer = model_data["vectorizer"]
label_encoder = model_data["label_encoder"]

# --- Page Configuration ---
st.set_page_config(page_title="Recipe Roadster", page_icon="🍔", layout="wide")

# --- Custom CSS ---
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Rock+Salt&display=swap');

:root {
    --primary-red: #FF0000;
    --neon-pink: #FF10F0;
    --retro-yellow: #FFD700;
    --dark-bg: #0D0D0D;
}

body {
    background-color: var(--dark-bg);
    color: white;
}

/* Retro Header */
.header {
    background: linear-gradient(180deg, #2F0B07 0%, #0D0D0D 100%);
    border-bottom: 3px solid var(--primary-red);
    padding: 1rem 0;
    text-align: center;
}

.title {
    font-family: 'Bebas Neue', cursive;
    font-size: 4rem;
    color: var(--retro-yellow);
    text-shadow: 2px 2px var(--primary-red);
    letter-spacing: 3px;
}

/* Navigation Menu */
.nav-menu {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin: 1rem 0;
}

.nav-item {
    font-family: 'Rock Salt', cursive;
    color: var(--neon-pink) !important;
    font-size: 1.2rem;
    cursor: pointer;
    transition: 0.3s;
}

.nav-item:hover {
    color: var(--retro-yellow) !important;
    transform: scale(1.1);
}

/* Recipe Card */
.recipe-card {
    background: #1A1A1A;
    border: 2px solid var(--primary-red);
    border-radius: 10px;
    padding: 2rem;
    margin: 2rem 0;
    box-shadow: 0 0 15px rgba(255,0,0,0.3);
}

/* Form Elements */
.stTextInput>div>div>input, .stTextArea>div>div>textarea {
    background: #2A2A2A !important;
    color: white !important;
    border: 1px solid var(--primary-red) !important;
    border-radius: 5px !important;
}

.stButton>button {
    background: var(--primary-red) !important;
    color: white !important;
    font-family: 'Bebas Neue', cursive !important;
    font-size: 1.5rem !important;
    border-radius: 5px !important;
    border: none !important;
    padding: 0.5rem 2rem !important;
    transition: 0.3s !important;
}

.stButton>button:hover {
    background: var(--retro-yellow) !important;
    color: var(--dark-bg) !important;
    transform: scale(1.05);
}

/* Gallery Section */
.gallery {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
}

.gallery-item {
    border-radius: 10px;
    overflow: hidden;
    transition: 0.3s;
}

.gallery-item:hover {
    transform: scale(1.05);
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Header Section ---
st.markdown("""
<div class="header">
    <div class="title">RECIPE ROADSTER</div>
    <div class="nav-menu">
        <span class="nav-item">🍔 HOME</span>
        <span class="nav-item">📜 MENU</span>
        <span class="nav-item">📸 GALLERY</span>
        <span class="nav-item">📞 CONTACT</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Main Content ---
with st.sidebar:
    st.markdown("""
    <div style="border-left: 3px solid var(--primary-red); padding: 1rem;">
        <h2 style="color: var(--neon-pink); font-family: 'Bebas Neue';">YOUR INGREDIENTS</h2>
        <p style="color: var(--retro-yellow);">What's in your kitchen?</p>
    """, unsafe_allow_html=True)
    ingredients = st.text_area(
        "Enter ingredients (comma-separated):",
        height=150,
        placeholder="e.g.: beef, cheese, tomatoes...",
        label_visibility="collapsed"
    )

# --- Prediction Section ---
col1, col2 = st.columns([2, 1])
with col1:
    if st.button("🚀 CREATE RECIPE!"):
        if not ingredients.strip():
            st.error("⚠️ Please enter some ingredients!")
        else:
            try:
                X_input = vectorizer.transform([ingredients]).toarray()
                y_pred = model.predict(X_input)
                recipe_index = np.argmax(y_pred, axis=1)
                predicted_recipe = label_encoder.inverse_transform(recipe_index)
                
                st.markdown(f"""
                <div class="recipe-card">
                    <h2 style="color: var(--retro-yellow); font-family: 'Bebas Neue';">🍳 TODAY'S SPECIAL</h2>
                    <div style="font-size: 2.5rem; text-align: center; margin: 1rem 0;">👩🍳</div>
                    <h1 style="text-align: center; color: var(--neon-pink);">{predicted_recipe[0]}</h1>
                    <div style="text-align: center; margin-top: 1rem;">
                        <span style="color: var(--primary-red);">➤</span> 
                        <span style="color: var(--retro-yellow);">Full recipe available in our cookbook!</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            except Exception as e:
                st.error(f"🔥 Error: {str(e)}")

with col2:
    st.markdown("""
    <div style="background: #1A1A1A; padding: 1.5rem; border-radius: 10px; border: 2px solid var(--primary-red);">
        <h3 style="color: var(--neon-pink); font-family: 'Bebas Neue';">DAILY SPECIAL</h3>
        <img src="https://images.unsplash.com/photo-1555939594-58d7cb561ad1?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80" 
             style="width: 100%; border-radius: 10px; margin: 1rem 0;">
        <p style="color: var(--retro-yellow);">Classic Beef Burger with Secret Sauce</p>
    </div>
    """, unsafe_allow_html=True)

# --- Gallery Section ---
st.markdown("""
<div class="gallery">
    <div class="gallery-item">
        <img src="https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80" 
             style="width: 100%; height: 200px; object-fit: cover;">
    </div>
    <div class="gallery-item">
        <img src="https://images.unsplash.com/photo-1565958011703-44f9829ba187?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80" 
             style="width: 100%; height: 200px; object-fit: cover;">
    </div>
    <div class="gallery-item">
        <img src="https://images.unsplash.com/photo-1482049016688-2d3e1b311543?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80" 
             style="width: 100%; height: 200px; object-fit: cover;">
    </div>
</div>
""", unsafe_allow_html=True)

# --- Contact Section ---
st.markdown("""
<div style="background: #1A1A1A; padding: 2rem; border-radius: 10px; margin-top: 2rem; border: 2px solid var(--primary-red);">
    <h2 style="color: var(--neon-pink); font-family: 'Bebas Neue';">📞 CONTACT THE CHEF</h2>
    <div style="display: grid; gap: 1rem; margin-top: 1rem;">
        <input type="text" placeholder="Name" style="padding: 0.5rem; background: #2A2A2A; border: 1px solid var(--primary-red); color: white; border-radius: 5px;">
        <input type="email" placeholder="Email" style="padding: 0.5rem; background: #2A2A2A; border: 1px solid var(--primary-red); color: white; border-radius: 5px;">
        <textarea placeholder="Message" style="padding: 0.5rem; background: #2A2A2A; border: 1px solid var(--primary-red); color: white; border-radius: 5px; height: 100px;"></textarea>
        <button style="background: var(--primary-red); color: white; padding: 0.5rem 2rem; border: none; border-radius: 5px; cursor: pointer;">SEND MESSAGE</button>
    </div>
</div>
""", unsafe_allow_html=True)
