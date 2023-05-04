from ntcore import NetworkTableInstance, EventFlags
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import time
import math
import numpy as np

ntinst = NetworkTableInstance.getDefault()

powerTable = ntinst.getTable('Power')
elbow = powerTable.getDoubleTopic("elbowPower").publish()
slide = powerTable.getDoubleTopic("slidePower").publish()

hitTable = ntinst.getTable('Hit')
right = hitTable.getBooleanTopic("rightClosed").publish()
left = hitTable.getBooleanTopic("leftClosed").publish()


ntinst.startClient4("wpilibpi")
ntinst.setServerTeam(4930)
ntinst.startDSClient()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
 
pTime = 0
cTime = 0

ax = plt.axes(projection="3d")
x_data = np.random.randint(0, 100, (500,))
y_data = np.random.randint(0, 100, (500,))
z_data = np.random.randint(0, 100, (500,))

ax.scatter(x_data, y_data, z_data)

 
while True:
    success, image = cap.read()
    imgRGB = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img = cv2.flip(imgRGB, 1)
    results = hands.process(img)

    landmark_5 = 0
    landmark_17 = 0

    landmark_9 = 0
    landmark_0 = 0

    landmark_12 = 0

    elbowPower = 0.0
    slidePower = 0.0

    rightClosed = False
    leftClosed = False

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
            
            cv2.circle(img, (xFollow, yFollow), 10, (255, 0, 0), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            elbowPower = -1 * (((yFollow - 0) / (720 - 0)) * 2 - 1) / 2 
            slidePower = -1 * (((xFollow - 0) / (1280 - 0 )) * 2 - 1) / 4

            if(handWidth < 100):
                rightClosed = True
                leftClosed = True
                elbowPower = 0.0
                slidePower = 0.0

            # cv2.line((640, 320), (xFollow, 320))
            # cv2.line((xFollow, 320), (xFollow, yFollow))



    
    text = "Elbow Power:"
    number = elbowPower
    cv2.putText(img, f"{text} {number: .2f}", (0, 60), cv2.FONT_HERSHEY_PLAIN, 2,
        (255, 255, 255), 3)
    
    text = "Slide Power:"
    number = slidePower
    cv2.putText(img, f"{text} {number: .2f}", (640, 60), cv2.FONT_HERSHEY_PLAIN, 2,
        (255, 255, 255), 3)
    
    if( rightClosed ):
        cv2.putText(img, "HIT", (640, 320), cv2.FONT_HERSHEY_PLAIN, 2,
        (0, 165, 255), 3)

    elbow.set(elbowPower)
    slide.set(slidePower)

    right.set(rightClosed)   
    left.set(leftClosed)
    

    if cv2.waitKey(1) == ord('q'):
        break
 
    cv2.imshow("Hand Tracker", img)
    plt.show()
    cv2.waitKey(1)