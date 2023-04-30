from ntcore import NetworkTableInstance

import cv2
import mediapipe as mp
import time
import math

ntinst = NetworkTableInstance.getDefault()

powerTable = ntinst.getTable('Power')
elbow = powerTable.getDoubleTopic("elbowPower").publish()
slide = powerTable.getDoubleTopic("slidePower").publish()

hitTable = ntinst.getTable('Hit')
right = hitTable.getBooleanTopic("rightClosed").publish()
left = hitTable.getBooleanTopic("leftClosed").publish()




cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
 
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
 
pTime = 0
cTime = 0
 
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    landmark_5 = 0
    landmark_17 = 0

    landmark_9 = 0
    landmark_0 = 0

    landmark_12 = 0

    elbowRectangle = (20, 0, 200, 720)
    slideRectangle = (600, 540, 1280, 720)

    elbowDistance = 0
    slideDistance = 0

    elbowPower = 0.0
    slidePower = 0.0

    elbowClosed = False
    slideClosed = False



    cv2.rectangle(img, (20, 0), (200, 720), (0, 0, 255), 5)
    cv2.line(img, (20, 360), (200, 360), (255, 255, 255), 2)

    cv2.rectangle(img, (600, 540), (1280, 720), (0, 0, 255), 5)
    cv2.line(img, (940, 540), (940, 720), (255, 255, 255), 2)

    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h) 

                if(id == 5):
                    landmark_5 = cx
                if(id == 17):
                    landmark_17 = cx

                if(id == 9):
                    landmark_9 = cy
                if(id == 0):
                    landmark_0 = cy

                if(id ==12):
                    landmark_12 = cy

            xWidth = int((landmark_17 - landmark_5) / 2)
            yWidth = int((landmark_0 - landmark_9) / 2)

            xFollow = landmark_5 + xWidth
            yFollow = landmark_9 + yWidth

            handWidth = int(landmark_0 - landmark_12)

            if(xFollow > elbowRectangle[0] and xFollow < elbowRectangle[2] and yFollow > elbowRectangle[1] and yFollow < elbowRectangle[3]):
                cv2.rectangle(img, (20, 0), (200, 720), (0, 255, 0), 5)
                cv2.line(img, (xFollow, yFollow), (110, 360), (255, 255, 255), 2)
                elbowDistance = math.sqrt((xFollow - 110)**2 + (yFollow - 360)**2)
                if( yFollow - 360 > 0 ):
                    elbowDistance = elbowDistance * -1
                elbowPower = (elbowDistance - 0) / (370 - 0 ) / 2
                if(handWidth < 100):
                    elbowClosed = True
                    elbowPower = 0.0



            if(xFollow > slideRectangle[0] and xFollow < slideRectangle[2] and yFollow > slideRectangle[1] and yFollow < slideRectangle[3]):
                cv2.rectangle(img, (600, 540), (1280, 720), (0, 255, 0), 5)
                cv2.line(img, (xFollow, yFollow), (940, 630), (255, 255, 255), 2)
                slideDistance = math.sqrt((xFollow - 940)**2 + (yFollow - 630)**2)
                if( xFollow - 940 < 0 ):
                    slideDistance = slideDistance * -1
                slidePower = (slideDistance - 0) / (350 - 0 ) / 2
                if(handWidth < 100):
                    slideClosed = True
                    slidePower = 0.0





            cv2.circle(img, (xFollow, yFollow), 10, (255, 0, 0), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.putText(img, "Elbow Controller", (0, 30), cv2.FONT_HERSHEY_PLAIN, 2,
        (255, 255, 255), 3)
    
    text = "Elbow Power:"
    number = elbowPower
    cv2.putText(img, f"{text} {number: .2f}", (0, 60), cv2.FONT_HERSHEY_PLAIN, 2,
        (255, 255, 255), 3)
    
    cv2.putText(img, "Slide Controller", (850, 540), cv2.FONT_HERSHEY_PLAIN, 2,
        (255, 255, 255), 3)
    
    text = "Slide Power:"
    number = slidePower
    cv2.putText(img, f"{text} {number: .2f}", (850, 570), cv2.FONT_HERSHEY_PLAIN, 2,
        (255, 255, 255), 3)
    
    if( elbowClosed and slideClosed ):
        cv2.putText(img, "HIT", (640, 320), cv2.FONT_HERSHEY_PLAIN, 2,
        (0, 165, 255), 3)

    elbow.set(elbowPower)
    slide.set(slidePower)

    right.set(elbowClosed)
    left.set(slideClosed)    
    

    if cv2.waitKey(1) == ord('q'):
        break
 
    cv2.imshow("Hand Tracker", img)
    cv2.waitKey(1)