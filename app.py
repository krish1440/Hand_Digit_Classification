from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import base64
import re
import cv2
import os
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='static')


model_path = 'models/improved_mnist_model.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found. Please run the training script first.")
model = tf.keras.models.load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

def preprocess_image(img):
    """
    Improved preprocessing to convert user-drawn digit to MNIST-like format (28x28).
    """
    img_array = np.array(img)

    # Convert to grayscale 
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Normalize 
    img_array = img_array.astype('float32') / 255.0  

    # ensure white background, black digit
    if np.mean(img_array) > 0.5:  
        img_array = 1.0 - img_array

    #Gaussian Blur 
    img_array = cv2.GaussianBlur(img_array, (5,5), 0)

    #detect the digit region
    contours, _ = cv2.findContours((img_array * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)

        # Add small padding
        padding = max(10, int(0.2 * max(w, h)))  
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_array.shape[1] - x, w + 2 * padding)
        h = min(img_array.shape[0] - y, h + 2 * padding)

        digit = img_array[y:y+h, x:x+w]
    else:
        digit = img_array 

    # Resize to 28x28
    digit_resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)

    # Ensure the digit is centered
    final_image = np.zeros((28, 28), dtype=np.float32)
    h, w = digit_resized.shape
    pad_x = (28 - w) // 2
    pad_y = (28 - h) // 2
    final_image[pad_y:pad_y+h, pad_x:pad_x+w] = digit_resized

    # Reshape for the model
    return final_image.reshape(1, 28, 28, 1)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data['image']
        image_data = re.sub('^data:image/png;base64,', '', image_data)
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes)).convert('L')
        
        processed_img = preprocess_image(img).reshape(1, 28, 28, 1)
        
        prediction = model.predict(processed_img)
        digit = np.argmax(prediction)
        confidence = float(prediction[0][digit]) * 100
        
        return jsonify({
            'digit': int(digit),
            'confidence': confidence,
            'probabilities': prediction[0].tolist()
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'digit': -1,
            'confidence': 0.0,
            'probabilities': [0.0] * 10
        }), 500

if __name__ == '__main__':
    app.run(debug=True)