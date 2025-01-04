import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Parameters
offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a blank white image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand region
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Check the shape of the cropped image
        imgCropShape = imgCrop.shape

        # Calculate the aspect ratio
        aspectRatio = h / w

        if aspectRatio > 1:  # Tall hand
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))  # Resize maintaining aspect ratio
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize  # Center the image horizontally
        else:  # Wide hand
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # Resize maintaining aspect ratio
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize  # Center the image vertically

        # Display cropped and white images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Display the original image
    cv2.imshow("Image", img)

    # Wait for key press
    key = cv2.waitKey(1)
    if key == 27:  # Exit when ESC is pressed
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
