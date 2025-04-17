import tensorflow as tf

# If needed, specify custom objects:
custom_objects = {"MeanSquaredError": tf.keras.losses.MeanSquaredError()}

model = tf.keras.models.load_model("age_model.h5", custom_objects=custom_objects)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# (Optional) Optimize model size or performance
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("age_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as 'age_model.tflite'")
