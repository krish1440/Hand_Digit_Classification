import tensorflow as tf
import os

def convert_model():
    model_path = 'models/improved_mnist_model.keras'
    tflite_path = 'models/model.tflite'
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    print("Loading model...")
    try:
        # Attempt to load the model
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optional: Optimizations
    # converter.optimizations = [tf.lite.Optimize.DEFAULT] 
    
    try:
        tflite_model = converter.convert()

        print(f"Saving to {tflite_path}...")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        print("Conversion complete.")
    except Exception as e:
        print(f"Error converting: {e}")

if __name__ == "__main__":
    convert_model()
