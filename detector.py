from __future__ import annotations
from importlib.resources import path
from pickle import FALSE
from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import numpy as np

# Variables
width, height = 800, 600
folderPath = 'Presentations'

cap = cv2.VideoCapture(1)
cap.set(3, width)
cap.set(4, height)

print('camera is live...')

# Import Images
pathImages = os.listdir(folderPath)
pathImages.sort()
pathImages.pop(0)

imgNumber = 0

hs, ws = int(120), int(180)

buttonPressed = False
counterFrame = 0
delayFrame = 10

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

annotations = []
key = -1
addAnnotation = True

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    hands, img = detector.findHands(img)

    xps, yps = 0, int(height / 2 + 80)

    img = cv2.line(img, (xps, yps),
                   (width, yps), (0, 255, 0), 5)

    if len(hands) > 0 and buttonPressed is False:
        hand = hands[0]
        handPos = hand['center']
        lmList = hand['lmList'][8]

        xVal = int(np.interp(lmList[0], [0, width//2], [0, width]))
        yVal = int(np.interp(lmList[1], [150, height-150], [0, height]))

        indexFinger = xVal, yVal

        fingers = detector.fingersUp(hand)

        if handPos[1] < yps:
            # Gesture 1 - Left
            if fingers == [1, 0, 0, 0, 0]:
                if imgNumber > 0:
                    buttonPressed = True
                    imgNumber -= 1
                    annotations = []
                    key = -1
                    addAnnotation = True

            # Gesture 2 - Right
            if fingers == [0, 0, 0, 0, 1]:
                if imgNumber < (len(pathImages) - 1):
                    buttonPressed = True
                    imgNumber += 1
                    annotations = []
                    key = -1
                    addAnnotation = True

        # Gesture 3 - Show On Pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            print(annotations)
            print(key)

        # Gesture 4 - Draw Pointer
        if fingers == [0, 1, 0, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            if addAnnotation:
                annotations.append([])
                key += 1
                addAnnotation = False

            print(key)
            annotations[key].append(indexFinger)
        else:
            addAnnotation = True

        # Gesture 5 - Ctrl+Z
        if fingers == [0, 1, 1, 1, 0]:
            if len(annotations) > 0:
                buttonPressed = True
                annotations.pop()
                addAnnotation = True
                key -= 1

    # Button Pressed
    if buttonPressed:
        counterFrame += 1
        if counterFrame >= delayFrame:
            counterFrame = 0
            buttonPressed = False

    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
                cv2.line(imgCurrent, annotations[i][j - 1],
                         annotations[i][j], (0, 0, 255), 12)

    # Adding Webcam Image On The Slides
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape

    imgCurrent[0:hs, w-ws:w] = imgSmall

    # Adding Text On The Slides
    cv2.putText(imgCurrent, f'Slide: {imgNumber + 1}',
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Image', img)
    cv2.imshow('Slides', imgCurrent)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
