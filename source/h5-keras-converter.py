import tensorflow as tf

# Load the .h5 model
h5_model_path = r"C:\Users\kishore l\sign-language-detector\model\best_model.h5"
model = tf.keras.models.load_model(h5_model_path)

# Save the model in TensorFlow's native format (.keras)
keras_model_path = r"C:\Users\kishore l\sign-language-detector\model\best_model.keras"
model.save(keras_model_path)
print(f"Model successfully converted and saved to: {keras_model_path}")
