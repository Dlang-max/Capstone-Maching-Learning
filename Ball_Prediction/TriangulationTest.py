######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras and Daniel Lang
# Date: 10/27/19 and 4/9/2023
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
from kalmanfilter import KalmanFilter
from ntcore import NetworkTableInstance, EventFlags
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import math

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/

kf = KalmanFilter()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#Setting up Networktables
ntinst = NetworkTableInstance.getDefault()

coordinatesTable = ntinst.getTable('Coordinates')

yCenter = coordinatesTable.getIntegerTopic("yCenter").publish()
xCenter = coordinatesTable.getIntegerTopic("xCenter").publish()
confidence = coordinatesTable.getIntegerTopic("confidence").publish()

foundTable = ntinst.getTable('Found')
found = foundTable.getBooleanTopic("ballFound").publish()

ntinst.startClient4("wpilibpi")
ntinst.setServerTeam(4930)
ntinst.startDSClient()

H_FOV = 43.60209
V_FOV = 33.3977099

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30, cameraNum=0, run=True ):

        self.run = run

        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(cameraNum)

        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
        ret = self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # disable auto-exposure
        ret = self.stream.set(cv2.CAP_PROP_EXPOSURE, -6)  # set the exposure to -4 (corresponds to a shutter speed of 1/1000)
        ret = self.stream.set(10, 400)


            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

X_FOCAL_LEN = (imW / 2) /  math.tan(math.radians(H_FOV / 2))
Y_FOCAL_LEN = (imH / 2) /  math.tan(math.radians(V_FOV / 2))

CAMERA_DISTANCE = 0.3175

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter_right = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    interpreter_left = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter_right = Interpreter(model_path=PATH_TO_CKPT)
    interpreter_left = Interpreter(model_path=PATH_TO_CKPT)

interpreter_right.allocate_tensors()
interpreter_left.allocate_tensors()


# Get model details
input_details_right = interpreter_right.get_input_details()
output_details_right = interpreter_right.get_output_details()
height = input_details_right[0]['shape'][1]
width = input_details_right[0]['shape'][2]

floating_model = (input_details_right[0]['dtype'] == np.float32)

input_details_left = interpreter_left.get_input_details()
output_details_left = interpreter_left.get_output_details()
height = input_details_left[0]['shape'][1]
width = input_details_left[0]['shape'][2]

floating_model = (input_details_left[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details_right[0]['name']


if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream_right = VideoStream(resolution=(imW,imH),framerate=30, cameraNum=0, run=True).start()
videostream_left = VideoStream(resolution=(imW,imH),framerate=30, cameraNum=2, run=True).start()

time.sleep(1)

def angles_from_center( x, y, top_left=True, degrees=True):

    if top_left:
        x = x - (imW / 2)
        y = (imH / 2) - y

    xtan = x / X_FOCAL_LEN
    ytan = y / Y_FOCAL_LEN

    xrad = math.atan(xtan)
    yrad = math.atan(ytan)

    if not degrees:
        return xrad, yrad
    
    return math.degrees(xrad),math.degrees(yrad)

def location(camera_distance, rcamera, lcamera, center=False, degrees=True):
        
    rxangle,ryangle = rcamera
    lxangle,lyangle = lcamera

    yangle = (ryangle+lyangle)/2

    if degrees:
            lxangle = math.radians(lxangle)
            rxangle = math.radians(rxangle)
            yangle  = math.radians( yangle)

    X, Z = intersection(camera_distance, lxangle, rxangle, degrees=False)
    Y = math.tan(yangle) * distance_from_origin(X, Z)

    if center:
            X -= camera_distance / 2
    
    D = math.sqrt( X * X + Y * Y + Z * Z )

    return X, Y, Z, D

def intersection(camera_distance, langle, rangle, degrees=False):
    if degrees:
            rangle = math.radians(rangle)
            langle = math.radians(langle)

    rangle = math.pi/2 + rangle
    langle = math.pi/2 - langle

    rtan = math.tan(rangle)
    ltan = math.tan(langle)

    Y = camera_distance / ( 1/ltan + 1/rtan )

    X = Y/ltan

    return X, Y

def distance_from_origin(*coordinates):
        return math.sqrt(sum([x**2 for x in coordinates]))

x_right = 0
y_right = 0

x_left = 0
y_left = 0

ballPositionsXYZ = np.array([(0, 0, 0)])


#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame_right = videostream_right.read()
    
    frame_left = videostream_left.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frameR = frame_right.copy()
    frame_rgb_right = cv2.cvtColor(frameR, cv2.COLOR_BGR2RGB)
    frame_resized_right = cv2.resize(frame_rgb_right, (width, height))
    input_data_right = np.expand_dims(frame_resized_right, axis=0)

    frameL = frame_left.copy()
    frame_rgb_left = cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)
    frame_resized_left = cv2.resize(frame_rgb_left, (width, height))
    input_data_left = np.expand_dims(frame_resized_left, axis=0)
    

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data_right = (np.float32(input_data_right) - input_mean) / input_std
        input_data_left = (np.float32(input_data_left) - input_mean) / input_std


    # Perform the actual detection by running the model with the image as input
    interpreter_right.set_tensor(input_details_right[0]['index'],input_data_right)
    interpreter_right.invoke()

    interpreter_left.set_tensor(input_details_left[0]['index'],input_data_left)
    interpreter_left.invoke()

    # Retrieve detection results
    boxes_right = interpreter_right.get_tensor(output_details_right[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes_right = interpreter_right.get_tensor(output_details_right[classes_idx]['index'])[0] # Class index of detected objects
    scores_right = interpreter_right.get_tensor(output_details_right[scores_idx]['index'])[0] # Confidence of detected objects

    boxes_left = interpreter_left.get_tensor(output_details_left[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes_left = interpreter_left.get_tensor(output_details_left[classes_idx]['index'])[0] # Class index of detected objects
    scores_left = interpreter_left.get_tensor(output_details_left[scores_idx]['index'])[0] # Confidence of detected objects

    

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores_right)):
        if ((scores_right[i] > min_conf_threshold) and (scores_right[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()

            yminR = int(max(1,(boxes_right[i][0] * imH)))
            xminR = int(max(1,(boxes_right[i][1] * imW)))
            ymaxR = int(min(imH,(boxes_right[i][2] * imH)))
            xmaxR = int(min(imW,(boxes_right[i][3] * imW)))

            WIDTH = xmaxR - xminR
            HEIGHT = ymaxR - yminR

            if WIDTH > 0:
                x_right = xminR + (WIDTH / 2)
                y_right = yminR + (HEIGHT / 2)

            print(WIDTH)

            cv2.rectangle(frameR, (xminR,yminR), (xmaxR,ymaxR), (10, 255, 0), 2)



            # Draw label
            object_name = labels[int(classes_right[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores_right[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(yminR, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frameR, (xminR, label_ymin-labelSize[1]-10), (xminR+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frameR, label, (xminR, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    cv2.line(frameR, (imW // 2, 0), (imW // 2, imH), (0, 255, 0), 2) # Draw white line
    cv2.line(frameR, (0, imH // 2), ( imW, imH // 2), (0, 255, 0), 2) # Draw white line

    for i in range(len(scores_left)):
        if ((scores_left[i] > min_conf_threshold) and (scores_left[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            yminL = int(max(1,(boxes_left[i][0] * imH)))
            xminL = int(max(1,(boxes_left[i][1] * imW)))
            ymaxL = int(min(imH,(boxes_left[i][2] * imH)))
            xmaxL = int(min(imW,(boxes_left[i][3] * imW)))

            WIDTH = xmaxL - xminL
            HEIGHT = ymaxL - yminL

            if WIDTH > 0:
                x_left = xminL + (WIDTH / 2)
                y_left = yminL + (HEIGHT / 2)
            
            cv2.rectangle(frameL, (xminL,yminL), (xmaxL,ymaxL), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes_left[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores_left[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(yminL, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frameL, (xminL, label_ymin-labelSize[1]-10), (xminL+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in


            cv2.putText(frameL, label, (xminL, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    cv2.line(frameL, (imW // 2, 0), (imW // 2, imH), (0, 255, 0), 2) # Draw white line
    cv2.line(frameL, (0, imH // 2), ( imW, imH // 2), (0, 255, 0), 2) # Draw white line

    # Draw framerate in corner of frame
    cv2.putText(frameL,'FPS: {0:.2f}'.format(frame_rate_calc),(0, 150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    frame_concat = cv2.hconcat([frameL, frameR])

    xrangle, yrangle = angles_from_center( x_right, y_right, top_left=True, degrees=True)
    xlangle, ylangle = angles_from_center( x_left, y_left, top_left=True, degrees=True)
    X, Y, Z, D = location( CAMERA_DISTANCE, (xrangle, yrangle), (xlangle, ylangle), center=True, degrees=True )

    ballPositionsXYZ = np.append(ballPositionsXYZ, (X, Y, Z))

    for pt in ballPositionsXYZ:
        predictedXY = kf.predict(pt[0], pt[1])
        predictedYZ = kf.predict(pt[1], pt[2])

    for i in range(10):
        predictedXY = kf.predict(predictedXY[0], predictedXY[1])
        predictedYZ = kf.predict(predictedYZ[0], predictedYZ[1])

        print((predictedXY[0], predictedYZ[0], predictedYZ[1]))  


    predictedXY.clear()
    predictedYZ.clear()

    if(len(ballPositionsXYZ) > 100):
        ballPositionsXYZ = np.array([(0, 0, 0)])
        
    ax.clear()
    ax.scatter(ballPositionsXYZ.reshape(-1, 2)[:, 0], ballPositionsXYZ.reshape(-1, 2)[:, 1], ballPositionsXYZ.reshape(-1, 2)[:, 2])
    plt.draw()
    plt.pause(0.01)

    text = 'X: {:3.1f}\nY: {:3.1f}\nZ: {:3.1f}\nD: {:3.1f}'.format(X,Y,Z,D )
    lineloc = 0
    lineheight = 30
    for t in text.split('\n'):
        lineloc += lineheight
        cv2.putText(frame_concat,
                    t,
                    (10,lineloc), # location
                    cv2.FONT_HERSHEY_PLAIN, # font
                    #cv2.FONT_HERSHEY_SIMPLEX, # font
                    1.5, # size
                    (0,255,0), # color
                    1, # line width
                    cv2.LINE_AA, #
                    False) #

    cv2.imshow('Object detector', frame_concat)


    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1



    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream_right.stop()
videostream_left.stop()

