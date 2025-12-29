# 🧠 AI Handwritten Digit Recognition

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=for-the-badge&logo=render&logoColor=white)](https://hand-digit-classification.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Backend-lightgrey?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

> A high-performance, web-based Deep Learning application that recognizes handwritten digits (0-9) in real-time. Built with **Flask**, **TensorFlow/Keras**, and a modern **Glassmorphism UI**.

---

## 🚀 Live Demo
Experience the application live: **[https://hand-digit-classification.onrender.com/](https://hand-digit-classification.onrender.com/)**

---

## ✨ Features

- **🎨 Interactive Drawing Canvas**: Smooth, responsive HTML5 canvas for drawing digits.
- **⚡ Real-time Inference**: Powered by a lightweight **TFLite** model for instant predictions.
- **📊 Probability Visualization**: Dynamic bar charts showing confidence scores for all digits (0-9).
- **💎 Premium UI/UX**: Modern dark-themed interface with glassmorphism effects and responsive design.
- **📱 Mobile Compatible**: Fully optimized for touch devices and desktops.

---

## 🛠️ Tech Stack

### Backend
- **Python 3.10+**: Core logic.
- **Flask**: Lightweight web server.
- **TensorFlow / Keras**: Model training and architecture.
- **TFLite**: Optimized model inference for production.
- **Gunicorn**: Production WSGI server.

### Frontend
- **HTML5 / CSS3**: Semantic structure and custom premium styling.
- **JavaScript (ES6+)**: Canvas logic and fetch API integration.
- **Chart.js**: Visualizing probability distributions.

---

## 📂 Project Structure

```text
├── model/
│   ├── train.py           # Training script with Data Augmentation
│   ├── mnist_model.keras  # Master Keras Model
│   └── mnist_model.tflite # Optimized Production Model
├── static/
│   ├── css/style.css      # Premium Styling
│   └── js/script.js       # Drawing & Interaction Logic
├── templates/
│   └── index.html         # Main Application Interface
├── app.py                 # Flask Application Entry Point
├── Procfile               # Render Deployment Configuration
├── requirements.txt       # Project Dependencies
└── README.md              # Project Documentation
```

---

## 🏃‍♂️ Run Locally

Clone the repository and run the application on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/krish1440/Hand_Digit_Classification.git
cd Hand_Digit_Classification
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
Open your browser and visit: `http://localhost:5000`

---

## 🧠 Model details

The model is a **Convolutional Neural Network (CNN)** trained on the MNIST dataset.
- **Architecture**: 3 Conv2D layers, MaxPolling, BatchNormalization, and Dropout.
- **Accuracy**: ~99.3% on test set.
- **Optimization**: Quantized and converted to TFLite for fast, CPU-based inference.

---

## 👨‍💻 Author

**Krish Chaudhary**

[![Portfolio](https://img.shields.io/badge/Portfolio-Visit%20Site-blueviolet?style=flat-square&logo=google-chrome&logoColor=white)](https://portfolio-krish-chaudhary.vercel.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github&logoColor=white)](https://github.com/krish1440/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/krish-chaudhary-krc8252)

---

