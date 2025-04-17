import cv2
import numpy as np
import tensorflow as tf
from tkinter import Tk, filedialog
import warnings
import os

# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TF logs
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# Load the TFLite model and allocate tensors.
model_path = "age_model.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Retrieve model input and output details.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
print("Model input shape:", input_shape)

def load_and_preprocess_image(image_path):
    """
    Loads an image, converts BGR to RGB, resizes it to fit the model input,
    normalizes pixel values, and adds a batch dimension.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to load.")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape[2], input_shape[1]))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_age(image_path):
    """
    Predicts the age from an image file using the TFLite model.
    """
    input_data = load_and_preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_age = output_data[0][0]
    return predicted_age

def choose_image_file():
    """
    Opens a file dialog to let the user select an image file.
    """
    print("->Choose an image to detect age...")
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    return file_path

if __name__ == "__main__":
    img_path = choose_image_file()
    if img_path:
        try:
            age = predict_age(img_path)
            print(f"->Predicted Age: {age:.2f} years")
        except Exception as e:
            print("Error during inference:", e)
    else:
        print("No image selected.")
