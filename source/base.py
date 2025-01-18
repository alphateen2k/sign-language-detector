import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300
counter = 0

# Define base folder for dataset
base_folder = r"C:\Users\kishore l\dataset_sign_language"
symbol = input("Enter the symbol (e.g., A, B, C) you want to capture: ").strip().upper()
folder = os.path.join(base_folder, symbol)

# Create folder if it doesn't exist
os.makedirs(folder, exist_ok=True)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size > 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:  
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))  
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize  
            else:  # Wide hand
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal)) 
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize  
            
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord("s"):  # Press 's' to save the image
        counter += 1
        # Save the image with the label (symbol) and image number
        file_path = os.path.join(folder, f"{symbol}_{counter}.jpg")
        cv2.imwrite(file_path, imgWhite)
        print(f"Saved: {file_path} | Count: {counter}")

    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
