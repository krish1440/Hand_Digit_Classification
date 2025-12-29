import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Setup & Data Loading
def load_data():
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize to [0, 1] and reshape to (28, 28, 1)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

# 2. Model Definition
def build_model(input_shape):
    print("Building model...")
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        
        # First Block
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Second Block
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Third Block
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense
        layers.Flatten(),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(10, activation="softmax")
    ])
    return model

# 3. Training
def train_model():
    (x_train, y_train), (x_test, y_test) = load_data()
    input_shape = (28, 28, 1)
    
    model = build_model(input_shape)
    model.summary()
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1
    )
    
    batch_size = 128
    epochs = 15 # Efficient number of epochs for demonstration
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    
    print("Starting training...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks
    )
    
    # Evaluate
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    
    return model

# 4. Save & Convert
if __name__ == "__main__":
    if not os.path.exists("model"):
        os.makedirs("model")
        
    trained_model = train_model()
    
    # Save standard Keras model
    keras_path = "model/mnist_model.keras"
    trained_model.save(keras_path)
    print(f"Model saved to {keras_path}")
    
    # Convert to TFLite
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(trained_model)
    
    # Optional: Quantization for further optimization (Drastically reduces size with minimal accuracy loss)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] 
    
    tflite_model = converter.convert()
    
    tflite_path = "model/mnist_model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"Optimized TFLite model saved to {tflite_path}")
    print(f"Keras model size: {os.path.getsize(keras_path) / 1024:.2f} KB")
    print(f"TFLite model size: {os.path.getsize(tflite_path) / 1024:.2f} KB")
