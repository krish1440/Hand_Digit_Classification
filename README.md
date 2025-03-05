# Handwritten Digit Recognition

This is a web-based Handwritten Digit Recognition application using a deep learning model trained on the **MNIST dataset**. Users can draw a digit (0-9) in the provided canvas, and the model will predict the digit with confidence scores and probability distribution.

## Features
- Interactive drawing canvas to input handwritten digits
- Predict button to classify the drawn digit
- Clear button to reset the canvas
- Displays predicted digit with confidence percentage
- Shows probability distribution for all digits (0-9)
- Responsive UI for mobile and desktop support

## Tech Stack
- **Frontend:** HTML, CSS
- **Backend:** Flask (Python)
- **Deep Learning Model:** TensorFlow/Keras trained on MNIST dataset

## Installation & Setup
### Clone the repository
```sh
git clone https://github.com/krish1440/Hand_Digit_Classification.git
cd Hand_Digit_Classification
```

### Install dependencies
Make sure you have Python installed (preferably 3.8+). Install the required dependencies:
```sh
pip install -r requirements.txt
```

### Run the application
```sh
python app.py
```
The application will be available at `http://127.0.0.1:5000/`

## How It Works
1. The user draws a digit in the provided canvas.
2. The drawn image is converted into a format suitable for the MNIST-trained model.
3. The image is sent to the backend via a POST request.
4. The deep learning model processes the image and predicts the digit.
5. The result and probability distribution are displayed on the frontend.

## Web App Live  
[Web App Live](https://web-production-ba08.up.railway.app/)


## License
This project is licensed under the MIT License.

