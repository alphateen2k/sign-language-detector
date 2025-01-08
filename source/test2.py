import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
import tensorflow as tf
import time
import os

# Function to load model and handle compatibility
def load_model(model_path):
    try:
        # Load the model using the updated TensorFlow function
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure the model is converted to a compatible TensorFlow version.")
        raise e
    return model

# Function for prediction
def get_prediction(model, img, img_size):
    # Preprocess image
    img_resized = cv2.resize(img, (img_size, img_size))
    img_normalized = img_resized / 255.0  # Normalize pixel values
    img_input = np.expand_dims(img_normalized, axis=0)  # Add batch dimension
    predictions = model.predict(img_input)
    return predictions

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Load the trained model
model_path = "model/keras_model.h5"
labels_path = "model/labels.txt"

# Ensure model file exists
if not os.path.exists(model_path) or not os.path.exists(labels_path):
    raise FileNotFoundError("Model or labels file not found. Ensure the paths are correct.")

model = load_model(model_path)

# Load labels
with open(labels_path, "r") as file:
    labels = [line.strip() for line in file.readlines()]

# Parameters
offset = 20
img_size = 300

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam.")
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure cropping coordinates stay within the image boundaries
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        # Crop the hand region
        img_crop = img[y1:y2, x1:x2]

        if img_crop.size > 0:  # Proceed only if cropped region is valid
            # Create a blank white image
            img_white = np.ones((img_size, img_size, 3), np.uint8) * 255

            aspect_ratio = h / w
            if aspect_ratio > 1:  # Tall hand
                scale = img_size / h
                new_width = math.ceil(scale * w)
                img_resized = cv2.resize(img_crop, (new_width, img_size))
                w_gap = math.ceil((img_size - new_width) / 2)
                img_white[:, w_gap:w_gap + new_width] = img_resized
            else:  # Wide hand
                scale = img_size / w
                new_height = math.ceil(scale * h)
                img_resized = cv2.resize(img_crop, (img_size, new_height))
                h_gap = math.ceil((img_size - new_height) / 2)
                img_white[h_gap:h_gap + new_height, :] = img_resized

            # Get predictions
            predictions = get_prediction(model, img_white, img_size)
            predicted_label = labels[np.argmax(predictions)]
            confidence = np.max(predictions)

            # Display results
            print(f"Prediction: {predicted_label}, Confidence: {confidence:.2f}")

            # Display cropped and white images
            cv2.imshow("ImageCrop", img_crop)
            cv2.imshow("ImageWhite", img_white)

    # Display the original image
    cv2.imshow("Image", img)

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
