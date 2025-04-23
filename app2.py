import os
import json
import requests
import joblib
import numpy as np
import time
import streamlit as st
from streamlit_lottie import st_lottie
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
category_dict = np.load('category_dict.npy', allow_pickle=True).item()

# -------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="NourishAI Pro | Chef System",
    page_icon="👨‍🍳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# 2. LOAD MODEL (CACHED)
# -------------------------------
import gdown

def load_recipe_csv():
    file_id = '1lE6zl-9dJKUnNu6CDuBvHPpGodKrGnoX'
    output = 'raw_recipes.npy'
    url = f'https://drive.google.com/uc?id={file_id}'

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    df = np.load('raw_recipes.npy', allow_pickle=True).item()
    return df
recipe_dict = load_recipe_csv()
# @st.cache_resource
# def load_model():
#     file_id = '1tR6_8S-yISRKZR2QJ6BjU3A3V5HHUCm7'
#     destination = 'recipe_model.pkl'

#     if not os.path.exists(destination):
#         URL = f'https://drive.google.com/uc?id={file_id}'
#         session = requests.Session()
#         response = session.get(URL, stream=True)
#         if 'Content-Disposition' in response.headers:
#             with open(destination, 'wb') as f:
#                 for chunk in response.iter_content(32768):
#                     f.write(chunk)

#     model_data = joblib.load(destination)
#     return {
#         "model": model_data["model"],
#         "vectorizer": model_data["vectorizer"],
#         "label_encoder": model_data["label_encoder"]
#     }

# model_data = load_model()
import gdown
import os
import pickle
import streamlit as st
@st.cache_resource
def load_image_model():
    url = "https://github.com/CarlAmine/EECE-490/releases/download/v1.0/490Image.pkl"
    destination = "490Image.pkl"
    expected_size = 687 * 1024 * 1024  # 687 MB in bytes

    def download_file_stream(url, destination):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(destination, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    # Check if file is already good
    if not os.path.exists(destination) or os.path.getsize(destination) < expected_size:
        st.write("Downloading model file from GitHub...")
        try:
            download_file_stream(url, destination)
        except Exception as e:
            raise RuntimeError(f"Download failed: {e}")

    # Final size check
    if os.path.getsize(destination) < expected_size:
        raise ValueError("Downloaded file is too small. Possibly corrupted.")

    # Load model
    with open(destination, "rb") as f:
        model = pickle.load(f)

    return model



svc_model = load_image_model()
import faiss
from sentence_transformers import SentenceTransformer
import pickle

# Add this initialization section after your other model loads
@st.cache_resource
def load_fastapi_components():
    # Download FAISS index and data
    faiss_url = "https://drive.google.com/uc?id=1hyZ5OaqGT7pBRdY33oDFldldmozEfC0m"
    data_url = "https://drive.google.com/uc?id=1lhzN-CF5SDXPQo0lHY_BcGz6hqhzDVEg"
    
    # Download files using gdown
    faiss_path = gdown.download(faiss_url, "recipes_index.faiss", quiet=True)
    data_path = gdown.download(data_url, "recipes_data.pkl", quiet=True)
    
    # Load components
    index = faiss.read_index("recipes_index.faiss")
    with open("recipes_data.pkl", "rb") as f:
        rag_texts = pickle.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    return index, rag_texts, model
# -------------------------------
# 3. SESSION STATE
# -------------------------------
if 'generate_clicked' not in st.session_state:
    st.session_state.generate_clicked = False
if 'ingredients' not in st.session_state:
    st.session_state.ingredients = ""

# -------------------------------
# 4. GLOBAL STYLING
# -------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root {
    --primary: #1E90FF;
    --bg: #FFFFFF;
    --card: #FFFFFF;
    --text: #1E90FF;
    --input: #FFFFFF;
    --border: #87CEFA;
}
body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Roboto', sans-serif;
}
.title-text {
    font-size: 6rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    color: #2E3C49;
}
.subtitle-text {
    font-size: 2rem;
    font-weight: 500;
    margin-bottom: 2rem;
    color: #2E3C49;
}
div[data-testid="stTextArea"] label {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: #2E3C49 !important;
}
.stTextArea textarea {
    font-size: 1.6rem;
    padding: 1.5rem;
    background: var(--input);
    border: 2px solid var(--border);
    border-radius: 12px;
    min-height: 200px;
    color: var(--text);
}
.stButton>button {
    font-size: 1.6rem;
    padding: 1rem 2rem;
    background: var(--primary);
    color: #FFF;
    border: none;
    border-radius: 12px;
    margin-top: 1.5rem;
    width: 100%;
}
.stButton>button:hover {
    background: #1A7AD9;
}
.recipe-card {
    background: var(--card);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 2px solid var(--border);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    color: var(--text);
}
.recipe-card h3 {
    font-size: 1.6rem;
    margin-bottom: 1rem;
    color: var(--text);
}
.big-text {
    font-size: 1.4rem;
    line-height: 1.6;
    color: var(--text);
}
.photo-container {
    width: 100%;
    padding-top: 100%;
    position: relative;
    margin: 0 auto;
}
.photo-container img {
    position: absolute;
    top: 0; left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 12px;
    border: 2px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 5. HEADER
# -------------------------------
st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h1 class="title-text">NOURISH AI</h1>
    <p class="subtitle-text">Professional Recipe Generator</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# 6. LOAD LOTTIE FILES
# -------------------------------
from PIL import Image
import numpy as np
import io

def preprocess_for_svc(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((15, 15))  # <-- use the size from your training
    img_array = np.array(img) / 255.0  # normalize if you trained this way
    flat_array = img_array.flatten().reshape(1, -1)  # make it 1D per image
    return flat_array
import re

def format_nutrition(nutrition_data):
    # Convert to string and clean
    nutrition_str = str(nutrition_data)
    
    # Remove all non-numeric characters except commas and decimals
    cleaned = re.sub(r'[^\d.,]', '', nutrition_str)
    
    # Split into individual number strings
    number_strings = [s for s in cleaned.split(',') if s]
    
    # Convert to floats, ignoring empty/invalid values
    numbers = []
    for num_str in number_strings:
        try:
            numbers.append(float(num_str))
        except ValueError:
            continue
    
    # Map to nutrition labels
    labels = ['Calories', 'total fat', 'total sugar', 
             'sodium', 'protein', 'saturated fat']
    
    nutrition_dict = {}
    if len(numbers) >= 6:
        # Use indices 0-4, then 6 for 7-element lists
        indices = [0, 1, 2, 3, 4, 6] if len(numbers) == 7 else range(6)
        
        for i, label in enumerate(labels):
            try:
                nutrition_dict[label] = numbers[indices[i]]
            except (IndexError, KeyError):
                nutrition_dict[label] = None
    else:
        # Return all None if insufficient data
        nutrition_dict = {label: None for label in labels}
    
    return nutrition_dict
def get_recipe_attributes(name):
    target = name.lower()  # Case-insensitive search
    # Get indices of rows where 'name' column contains the target substring
    matching_indices = [
        index 
        for index, recipe_name in recipe_dict['name'].items() 
        if target in recipe_name.lower()
    ]
    
    if not matching_indices:
        return {"error": "No recipe found with that name"}
    
    first_match_index = matching_indices[0]
    
    return {
        'minutes': recipe_dict['minutes'][first_match_index],
        'ingredients': recipe_dict['ingredients'][first_match_index],
        'steps': recipe_dict['steps'][first_match_index],
        'nutrition': recipe_dict['nutrition'][first_match_index]
    }
# Helper function to format ingredients
def format_ingredients(raw_ingredients):
    # Join all characters into a single string
    ingredients_str = ''.join(raw_ingredients)
    # Split into individual ingredients using commas
    return [ing.strip(" '\"") for ing in ingredients_str.split(',') if ing.strip()]
def load_lottie_file(filename):
    path = os.path.join(r"C:\Users\AUB\Documents\GitHub\EECE-490", filename)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            return None
    return None

lottie_data = load_lottie_file("Animation.json")
lottie_data_2 = load_lottie_file("Animation1.json")

# -------------------------------
# 7. SHOW LOTTIE ANIMATION
# -------------------------------
def show_lottie_animation(animation_data, speed=1, height=300, width=300, key_suffix=""):
    if animation_data:
        st.markdown(
            f"""<div style="background-color:#FFFFFF; padding:1rem; border-radius:12px; text-align:center;">""",
            unsafe_allow_html=True
        )
        st_lottie(
            animation_data,
            speed=speed,
            reverse=False,
            loop=True,
            quality="high",
            height=height,
            width=width,
            key=f"anim_{key_suffix}"
        )
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 8. LAYOUT
# -------------------------------
col_input, col_anim, col_output = st.columns([1, 0.8, 1])

# INPUT + IMAGES
with col_input:
    uploaded_file = st.file_uploader("OR Upload an Image of the Dish:", type=["jpg", "jpeg", "png"], key="image_upload")
    ingredients = st.text_area(
        "ENTER INGREDIENTS (COMMA SEPARATED):",
        height=200,
        placeholder="e.g., chicken breast, garlic, olive oil, fresh basil...",
        key="main_input"
    )
    if st.button("🔍 GENERATE RECIPES", key="main_button"):
        if not ingredients.strip():
            st.error("Please enter ingredients to continue", icon="⚠️")
            st.session_state.generate_clicked = False
        else:
            st.session_state.generate_clicked = True
            st.session_state.ingredients = ingredients

    st.markdown("<hr style='border: none; height: 2px; background: var(--border);'>", unsafe_allow_html=True)

    image_urls = [
        "https://images.unsplash.com/photo-1546069901-ba9599a7e63c",
        "https://images.unsplash.com/photo-1512621776951-a57141f2eefd",
        "https://images.unsplash.com/photo-1490645935967-10de6ba17061",
        "https://images.unsplash.com/photo-1504674900247-0877df9cc836"
    ]

    for i in range(0, 4, 2):
        row = st.columns(2)
        for j in range(2):
            with row[j]:
                st.markdown(f"""
                <div class="photo-container">
                    <img src="{image_urls[i + j]}" alt="Photo {i + j + 1}">
                </div>
                """, unsafe_allow_html=True)

# ANIMATIONS
with col_anim:
    if st.session_state.generate_clicked and lottie_data:
        show_lottie_animation(lottie_data, key_suffix="1")
    if st.session_state.generate_clicked and lottie_data_2:
        show_lottie_animation(lottie_data_2, key_suffix="2")

# OUTPUT RECIPES
with col_output:
    if st.session_state.generate_clicked:
        with st.spinner("Analyzing ingredients and generating recipes..."):
            try:
                # Load components if not already loaded
                if 'faiss_components' not in st.session_state:
                    index, rag_texts, model = load_fastapi_components()
                    st.session_state.faiss_components = (index, rag_texts, model)
                else:
                    index, rag_texts, model = st.session_state.faiss_components

                # Perform the FAISS search
                q_embed = model.encode([ingredients], convert_to_numpy=True)
                faiss.normalize_L2(q_embed)
                scores, ids = index.search(q_embed, k=3)

                # Format results
                results = []
                for i, idx in enumerate(ids[0]):
                    recipe = rag_texts[idx]
                    match = round(scores[0][i] * 100, 2)
                    results.append(f"✅ Match: {match}%\n\n{recipe}")

                output = "\n\n\n".join(results)
                
                st.success("✅ Recipes generated:")
                st.markdown(f"<pre style='background:#f9f9f9; padding:1rem'>{output}</pre>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error generating recipes: {e}")
    if uploaded_file is not None:
        try:
            with st.spinner("Analyzing image and identifying dish..."):
                img_array = preprocess_for_svc(uploaded_file.read())
                prediction = svc_model.predict(img_array)
                predicted_class = category_dict[prediction[0]]
                predicted_class = predicted_class.replace('_',' ')
                
                # Display prediction
                st.markdown(f"""
                <div class="recipe-card">
                    <h3>📷 Dish Identified from Image</h3>
                    <p class="big-text">🍽️ <strong>{predicted_class}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Get recipe attributes using the predicted class
                recipe_data = get_recipe_attributes(predicted_class)
                
                # Check if recipe found
                if 'error' not in recipe_data:
                    # Format the components
                    ingredients = format_ingredients(recipe_data['ingredients'])
                    steps = ''.join([step.strip("'") for step in recipe_data['steps']])
                    nutrition = format_nutrition(recipe_data['nutrition'])
                    
                    # Display recipe details
                    st.markdown(f"""
                    <div class="recipe-card">
                        <h3>📝 Recipe Details</h3>
                        <p>⏱ Cooking Time: <strong>{recipe_data['minutes']} minutes</strong></p>
                        <p>🥕 Ingredients: <strong>{ingredients}</strong></p>
                        <pre>👩🍳 Steps:\n{steps}</pre>
                        <pre>📊 Nutrition:\n{nutrition}</pre>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"No recipe found for {predicted_class}", icon="⚠️")
    
        except Exception as e:
            st.error(f"Error: {str(e)}", icon="🛑")
# -------------------------------
# 9. SUPPRESS TENSORFLOW WARNINGS
# -------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

