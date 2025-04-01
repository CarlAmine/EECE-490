from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import tensorflow as tf

app = Flask(__name__, template_folder="templates")

# --- Load Preprocessor and Model ---
with open('recipe_preprocessor.pkl', 'rb') as f:
    model_data = pickle.load(f)
vectorizer = model_data['vectorizer']
label_encoder = model_data['label_encoder']

# Load the model from H5 file
model = tf.keras.models.load_model('model.h5')  # Update filename

# --- Serve HTML Page ---
@app.route('/')
def home():
    return render_template('490.html')

# --- Prediction API ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        ingredients = request.form.get('ingredients', '').strip()
        if not ingredients:
            return jsonify({'error': 'No ingredients provided'})

        # Process input correctly
        X_input = vectorizer.transform([ingredients]).toarray()

        y_pred = model.predict(X_input)
        recipe_index = np.argmax(y_pred, axis=1)
        predicted_recipe = label_encoder.inverse_transform(recipe_index)

        return jsonify({'recipe': predicted_recipe[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)