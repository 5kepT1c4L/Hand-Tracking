import mediapipe as mp
import cv2 as cv
import time 

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

current_time = 0
past_time = 0

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: # handLms = all the 21 landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            for id, hand_landmarks in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(hand_landmarks.x * w), int(hand_landmarks.y * h)
                print(f"Landmark {id}: {cx}, {cy}")
                cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

    current_time = time.time()
    fps = 1/(current_time-past_time)
    past_time = current_time

    cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 2)
    cv.imshow("Image", img)
    cv.waitKey(1)