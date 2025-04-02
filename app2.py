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

# --- Model Download & Loading ---
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
st.set_page_config(page_title="NourishWise", page_icon="👩🍳", layout="wide")

# --- Custom CSS ---
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@700&family=Patrick+Hand&display=swap');

:root {
    --main-color: #FF6B6B;
    --secondary-color: #FFD700;
    --background: #FFF9F0;
}

body {
    background: var(--background);
}

h1 {
    color: var(--main-color);
    font-family: 'Comic Neue', cursive;
    text-shadow: 2px 2px var(--secondary-color);
    text-align: center;
    font-size: 3.5rem !important;
}

.main-container {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin: 1rem 0;
}

.sidebar-container {
    background: #FFF3E0 !important;
    border-radius: 15px !important;
    border: 3px solid #FFD1A4 !important;
    padding: 1.5rem !important;
}

.stTextArea textarea {
    border-radius: 10px !important;
    border: 2px solid #FFB347 !important;
    font-family: 'Patrick Hand', cursive !important;
    font-size: 1.2rem !important;
}

.stButton > button {
    background: var(--main-color);
    font-family: 'Comic Neue', cursive;
    font-size: 1.5rem !important;
    border-radius: 15px !important;
    padding: 12px 30px !important;
    transition: transform 0.2s;
    width: 100%;
}

.recipe-card {
    background: #FFF8E1;
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 2px dashed #FFB347;
}

.fun-fact {
    background: #E3F2FD;
    border-left: 5px solid #64B5F6;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 8px;
}

.cooking-tip {
    background: #FFEBEE;
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
}

footer {
    background: #FFE0B2;
    padding: 1.5rem;
    border-radius: 15px;
    margin-top: 2rem;
    text-align: center;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Header Section ---
st.markdown("<h1>👨👩👧👦 Family Recipe Finder</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p style='font-family: "Patrick Hand", cursive; font-size: 1.4rem; color: #666;'>
        Turn your kitchen ingredients into fun family meals! 🍳🌈
    </p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Input ---
with st.sidebar:
    st.markdown("""
    <div class='sidebar-container'>
        <h2 style='color: #FF6B6B;'>🧺 Your Ingredients</h2>
        <p style='font-family: "Patrick Hand";'>What's in your kitchen today?</p>
    """, unsafe_allow_html=True)
    
    ingredients = st.text_area(
        "Enter ingredients (comma-separated):",
        height=150,
        placeholder="e.g.: eggs, milk, flour...",
        label_visibility="collapsed"
    )
    
    st.markdown("""
        <div style='margin-top: 1.5rem; font-family: "Patrick Hand";'>
        <p>🎉 Pro tips:</p>
        <ul>
            <li>Try 3-5 ingredients</li>
            <li>Be specific (e.g., 'chicken' vs 'meat')</li>
            <li>Have fun experimenting!</li>
        </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Main Content ---
col1, col2 = st.columns([3, 1])

with col1:
    with st.container():
        st.markdown("<div class='main-container'>", unsafe_allow_html=True)
        st.markdown("### 🍽️ Today's Featured Recipe")
        st.image("https://images.unsplash.com/photo-1567620905732-2d1ec7ab7445?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
                 use_container_width=True)
        st.markdown("""
        <div class='cooking-tip'>
            <h4 style='color: #FF6B6B;'>👩🍳 Chef's Special: Rainbow Wraps</h4>
            <p>Perfect for getting kids to eat veggies! Let them choose their colors!</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("### 📚 Cooking School")
    st.markdown("""
    <div class='fun-fact'>
        <h4>🥄 Did You Know?</h4>
        <p>Mixing ingredients by hand helps kids develop fine motor skills!</p>
    </div>
    """, unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1589984662646-e7b2e4962fce?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
             use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Prediction Section ---
if st.button("✨ Let's Create Magic!", use_container_width=True):
    if not ingredients.strip():
        st.error("⛔ Oops! We need some ingredients to work our magic!")
    else:
        try:
            X_input = vectorizer.transform([ingredients]).toarray()
            y_pred = model.predict(X_input)
            recipe_index = np.argmax(y_pred, axis=1)
            predicted_recipe = label_encoder.inverse_transform(recipe_index)
            
            st.markdown(f"""
            <div class='recipe-card'>
                <h3 style='color: #FF6B6B;'>🎉 Ta-da! We Recommend...</h3>
                <div style='font-size: 2.5rem; text-align: center; margin: 1rem 0;'>🧁</div>
                <h2 style='text-align: center;'>{predicted_recipe[0]}</h2>
                
                <div class='fun-fact' style='margin-top: 1.5rem;'>
                    <h4>👨👩👧👦 Family Challenge:</h4>
                    <p>Make it a team effort! Assign roles:<br>
                    - Little chefs: Mixing & decorating<br>
                    - Big chefs: Measuring & cooking</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.balloons()
            
            # Additional Recipe Suggestions
            st.markdown("""
            <div class='main-container' style='margin-top: 2rem;'>
                <h3>🍴 More Fun Ideas</h3>
                <div class='cooking-tip'>
                    <p>Try these variations with your ingredients:</p>
                    <ul>
                        <li>Make it colorful - add food coloring!</li>
                        <li>Create fun shapes with cookie cutters</li>
                        <li>Host a mini cooking competition</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Oh no! Our recipe book got messy! Please try again.")

# --- Footer ---
st.markdown("""
<footer>
    <p style='font-family: "Comic Neue"; font-size: 1.1rem; color: #666;'>
        🍳 Made with love by Family Kitchen Friends 🍪<br>
        Let's create delicious memories together!
    </p>
</footer>
""", unsafe_allow_html=True)
