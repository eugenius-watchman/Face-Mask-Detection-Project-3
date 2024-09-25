Face Mask Detection System

This project implements a real-time face mask detection system using a deep learning model with TensorFlow and OpenCV. The system detects faces in video streams and predicts whether individuals are wearing masks, not wearing masks, or wearing improper masks.
Features

    Face Detection: Uses a pre-trained model to detect faces in a video stream.
    Mask Classification: Classifies detected faces into three categories: "With Mask," "No Mask," and "Improper Mask."
    Real-time Processing: Processes video frames in real-time to provide instant feedback.

Requirements

    Python 3.x
    TensorFlow
    OpenCV
    imutils
    NumPy

Installation

    Clone the repository:

    bash

git clone https://github.com/eugenius-watchman/Face-Mask-Detection-Project-3.git

Navigate to the project directory:

bash

cd restaurant_app

Install the required packages:

bash

    pip install -r requirements.txt

Model Files

Ensure you have the following model files in the face_detector directory:

    deploy.prototxt.txt
    res10_300x300_ssd_iter_140000.caffemodel
    mask_detector.model

These models are essential for face detection and mask classification.
Usage

To run the face mask detection system, execute the following Python script:

python

# Import necessary libraries
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

# Load the serialized face detector model
prototxtPath = "face_detector/deploy.prototxt.txt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the face mask detector model
maskNet = load_model("mask_detector.model")

# Initialize the video stream
vs = VideoStream(src=0).start()

# Main loop for detection
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    
    # (Optional) Apply various blurring techniques for image preprocessing
    ...
    
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    
    # Loop over detected face locations and predictions
    ...
    
    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()

Important Note:

    The program will display a window showing the video feed from your webcam. It will label detected faces based on the mask status.

Results

The output will show bounding boxes around detected faces with labels indicating whether they are wearing a mask, not wearing a mask, or wearing an improper mask, along with the corresponding probabilities.
License

This project is licensed under the MIT License.
