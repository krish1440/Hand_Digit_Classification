import os
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

# Load TFLite Model
MODEL_PATH = "model/mnist_model.tflite"
interpreter = None

def load_model():
    global interpreter
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        print("TFLite Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

load_model()

def preprocess_image(image_data):

    img_bytes = base64.b64decode(image_data.split(',')[1])
    img = Image.open(io.BytesIO(img_bytes)).convert('L')  # Grayscale
    
 
    img = img.resize((28, 28))
    
    # match MNIST (white digits on black background).
    img_array = np.array(img)
    img_array = 255 - img_array
    
    # Normalize to [0, 1]
    img_array = img_array.astype("float32") / 255.0

    # Reshape for model (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=0) # Batch dimension
    img_array = np.expand_dims(img_array, axis=-1) # Channel dimension
    
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not interpreter:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        processed_input = preprocess_image(image_data)
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], processed_input)
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        probabilities = output_data[0]
        
        prediction = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        
        return jsonify({
            'digit': prediction,
            'confidence': confidence,
            'probabilities': probabilities.tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

