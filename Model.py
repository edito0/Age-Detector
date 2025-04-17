import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('age_model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open('age_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite model saved as age_model.tflite")
