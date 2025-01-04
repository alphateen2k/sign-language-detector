import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
import time
import os

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Parameters
offset = 20
imgSize = 300
counter = 0

# Define the base folder for saving images
base_folder = r"C:\Users\kishore l\sign-language-detector\data"

# Ask the user to specify the symbol (e.g., A, B, C) to save images for
symbol = input("Enter the symbol (e.g., A, B, C) you want to capture: ").strip().upper()
folder = os.path.join(base_folder, symbol)

# Create the folder if it doesn't exist
os.makedirs(folder, exist_ok=True)

while True:
    success, img = cap.read()
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

            # Display cropped and white images
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    # Display the original image
    cv2.imshow("Image", img)

    # Wait for key press
    key = cv2.waitKey(1)
    if key == ord("s"):  # Save the image when 's' is pressed
        counter += 1
        file_path = os.path.join(folder, f"Image_{time.time()}.jpg")
        cv2.imwrite(file_path, imgWhite)
        print(f"Saved: {file_path} | Count: {counter}")

    if key == 27:  # Exit when ESC is pressed
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
