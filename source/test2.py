import cv2
import numpy as np
import time
import os
import keras
import math
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector

# Load the pre-trained model (replace with your model path)
model = keras.models.load_model(r"C:\Users\kishore l\sign-language-detector\model\keras_model.h5")

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Parameters
imgSize = 300

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure cropping coordinates stay within the image boundaries
        y1 = max(0, y - 20)
        y2 = min(img.shape[0], y + h + 20)
        x1 = max(0, x - 20)
        x2 = min(img.shape[1], x + w + 20)

        # Crop the hand region
        imgCrop = img[y1:y2, x1:x2]

        # Proceed only if the cropped region is valid
        if imgCrop.size > 0:
            # Create a blank white image
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Check the shape of the cropped image
            imgCropShape = imgCrop.shape

            # Calculate the aspect ratio
            aspectRatio = h / w

            if aspectRatio > 1:  # Tall hand
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))  # Resize maintaining aspect ratio
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize  # Center the image horizontally
            else:  # Wide hand
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # Resize maintaining aspect ratio
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize  # Center the image vertically

            # Prepare image for model prediction
            imgWhite = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)  # Convert to RGB
            imgWhite = np.expand_dims(imgWhite, axis=0) / 255.0  # Normalize and reshape for model input

            # Predict the label
            predictions = model.predict(imgWhite)
            label_idx = np.argmax(predictions)  # Get the index of the highest probability

            # Map index to label (assuming labels are A, B, C)
            labels = ['A', 'B', 'C']
            predicted_label = labels[label_idx]

            # Display prediction on the image
            cv2.putText(img, f"Prediction: {predicted_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Predicted Image", imgWhite[0])

    # Display the original image
    cv2.imshow("Image", img)

    # Wait for key press
    key = cv2.waitKey(1)
    if key == 27:  # Exit when ESC is pressed
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
