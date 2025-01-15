import cv2
import numpy as np
import math
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector

# Paths to the model and labels
model_path = r"C:\Users\kishore l\sign-language-detector\model\sign_language_model.keras"
model = tf.keras.models.load_model(model_path)

# Class labels (ensure they match the training labels)
labels = ["A", "B"]

# Webcam and Hand Detector Initialization
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Parameters
offset = 20
img_size = 300

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure the bounding box is within the frame
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        # Crop the hand region
        img_crop = img[y1:y2, x1:x2]

        # Ensure the cropped image has valid dimensions
        if img_crop.size > 0:
            img_white = np.ones((img_size, img_size, 3), np.uint8) * 255

            # Aspect ratio and resizing logic
            aspect_ratio = h / w
            if aspect_ratio > 1:  # Tall hand
                scale = img_size / h
                w_cal = math.ceil(scale * w)
                img_resized = cv2.resize(img_crop, (w_cal, img_size))
                w_gap = (img_size - w_cal) // 2
                img_white[:, w_gap:w_cal + w_gap] = img_resized
            else:  # Wide hand
                scale = img_size / w
                h_cal = math.ceil(scale * h)
                img_resized = cv2.resize(img_crop, (img_size, h_cal))
                h_gap = (img_size - h_cal) // 2
                img_white[h_gap:h_cal + h_gap, :] = img_resized

            # Normalize and expand dimensions
            img_white = img_white / 255.0
            img_white = np.expand_dims(img_white, axis=0)

            # Prediction
            predictions = model.predict(img_white)
            class_idx = np.argmax(predictions[0])
            predicted_label = labels[class_idx]
            confidence = predictions[0][class_idx]

            # Display the result
            cv2.putText(img, f'{predicted_label} ({confidence:.2f})', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the original webcam feed
    cv2.imshow("Sign Language Detector", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
