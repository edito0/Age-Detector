import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

def create_model(input_shape=(64, 64, 3)):
    """
    Creates a simple Convolutional Neural Network for age regression.
    Adjust the architecture based on your dataset and performance needs.
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        # Output layer with a single neuron for regression (predicts a continuous age value)
        Dense(1)
    ])
    
    # Compile the model with Mean Squared Error loss and mean absolute error as a metric.
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    # -------------------------
    # DATA PREPARATION SECTION
    # -------------------------
    # For demonstration, we create a synthetic dataset.
    # In a practical scenario, load your dataset and preprocess images (resizing, normalization, etc.).
    num_samples = 500
    # Create random images of size 64x64 and random age values between 0 and 100.
    X = np.random.rand(num_samples, 64, 64, 3)
    y = np.random.randint(0, 101, size=(num_samples,))  # Age labels between 0 and 100

    # -------------------------
    # MODEL CREATION & TRAINING
    # -------------------------
    model = create_model(input_shape=(64, 64, 3))
    print(model.summary())
    
    # Train the model (for demonstration, only a few epochs).
    history = model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)
    
    # -------------------------
    # SAVE THE MODEL
    # -------------------------
    # Save the trained model in the HDF5 format as "age_model.h5"
    model.save('age_model.h5')
    print("Model saved as 'age_model.h5'")
