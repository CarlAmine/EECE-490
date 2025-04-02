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

# --- Model Loading ---
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

body {
    background: linear-gradient(135deg, #FFF3E0 0%, #FFE4E1 100%);
    background-attachment: fixed;
}

h1 {
    color: #FF6B6B;
    font-family: 'Comic Neue', cursive;
    text-shadow: 2px 2px #FFD700;
    text-align: center;
    font-size: 3rem !important;
    margin-bottom: 0.5rem !important;
}

.ingredient-box {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    margin: 2rem auto;
    max-width: 800px;
    box-shadow: 0 8px 20px rgba(255,107,107,0.1);
    border: 3px solid #FFB347;
}

.stTextArea textarea {
    border-radius: 15px !important;
    border: 2px solid #FFB347 !important;
    font-family: 'Patrick Hand', cursive !important;
    font-size: 1.2rem !important;
    transition: all 0.3s ease !important;
}

.stTextArea textarea:focus {
    border-color: #FF6B6B !important;
    box-shadow: 0 0 10px rgba(255,107,107,0.2) !important;
}

.recipe-card {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    margin: 2rem 0;
    box-shadow: 0 8px 25px rgba(255,107,107,0.1);
    transition: transform 0.3s ease;
}

.recipe-card:hover {
    transform: translateY(-5px);
}

.stButton > button {
    background: #FF6B6B !important;
    color: white !important;
    font-family: 'Comic Neue', cursive !important;
    font-size: 1.5rem !important;
    border-radius: 15px !important;
    padding: 12px 30px !important;
    transition: all 0.3s ease !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(255,107,107,0.3) !important;
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(255,107,107,0.4) !important;
}

.stButton > button:active {
    transform: scale(0.95);
}

.image-hover {
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    border-radius: 15px;
}

.image-hover:hover {
    transform: scale(1.03);
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}

.fun-fact {
    background: #FFF8F0;
    border-left: 5px solid #FF6B6B;
    padding: 1.5rem;
    margin: 2rem 0;
    border-radius: 15px;
}

footer {
    background: white;
    padding: 2rem;
    border-radius: 20px;
    margin-top: 3rem;
    text-align: center;
    box-shadow: 0 -8px 20px rgba(255,107,107,0.05);
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

# --- Ingredients Input ---
with st.container():
    st.markdown("<div class='ingredient-box'>", unsafe_allow_html=True)
    ingredients = st.text_area(
        "### 🧺 Enter Your Ingredients",
        height=150,
        placeholder="Type your ingredients here (comma-separated)...\nExample: chicken, broccoli, rice, garlic...",
        help="Let's see what delicious meal we can create!"
    )
    st.markdown("</div>", unsafe_allow_html=True)

# --- Main Content ---
col1, col2 = st.columns([2, 1])

with col1:
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
                    <h2 style='color: #FF6B6B; text-align: center;'>🎉 Ta-da! We Recommend...</h2>
                    <div style='font-size: 3rem; text-align: center; margin: 1rem 0;'>🍲</div>
                    <h1 style='text-align: center; color: #666;'>{predicted_recipe[0]}</h1>
                    
                    <div class='fun-fact'>
                        <h3 style='color: #FF6B6B;'>👨👩👧👦 Family Cooking Challenge!</h3>
                        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;'>
                            <div style='padding: 1rem; background: #FFF0F0; border-radius: 15px;'>
                                <h4>👶 Little Chefs</h4>
                                <p>• Mixing ingredients<br>• Decorating<br>• Choosing colors</p>
                            </div>
                            <div style='padding: 1rem; background: #FFF0F0; border-radius: 15px;'>
                                <h4>👩🍳 Big Chefs</h4>
                                <p>• Measuring<br>• Cooking<br>• Time management</p>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            except Exception as e:
                st.error(f"❌ Oh no! Our recipe book got messy! Please try again.")

with col2:
    st.markdown("""
    <div class='recipe-card'>
        <h3 style='color: #FF6B6B;'>🍴 Today's Featured Recipe</h3>
        <img src='https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80' 
             class='image-hover' style='width: 100%; border-radius: 15px; margin: 1rem 0;'>
        <h4 style='color: #666;'>Rainbow Veggie Stir-Fry</h4>
        <p style='color: #888;'>A colorful mix of fresh vegetables in a tangy sauce</p>
    </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<footer>
    <p style='font-family: "Comic Neue"; font-size: 1.1rem; color: #666;'>
        🍳 Made with ❤️ by Family Kitchen Friends 🍪<br>
        Let's create delicious memories together!
    </p>
</footer>
""", unsafe_allow_html=True)
