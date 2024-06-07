from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import os

app = Flask(__name__)
CORS(app)

# Check if model and tokenizer files exist
model_path = 'phishing_detection_model.keras'
tokenizer_path = 'tokenizer.pkl'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

# Load the trained model
model = load_model(model_path)

# Load the tokenizer
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    features = tokenizer.texts_to_matrix([email_text], mode='tfidf')
    prediction = model.predict(features)
    result = 'phishing' if prediction[0][0] > 0.5 else 'not phishing'
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
