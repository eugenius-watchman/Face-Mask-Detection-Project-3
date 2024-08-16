from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
    # Grab the dimensions of the frame and construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    
    # Pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)
    
    # Initialize the list of faces, their corresponding locations, and the list of predictions
    # from face mask network
    faces = []
    locs = []
    preds = []
    
    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (probability) associated with the detection
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
        
            # Ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        
            # Extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224,
            # and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
        
            # Add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))
            
    # Only make predictions if at least one face is detected
    if len(faces) > 0:
        # For faster inference, make batch predictions on all faces at the same time rather than
        # one-by-one predictions in the above for loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
        
    # Return a 2-tuple of the face locations and their corresponding locations
    return (locs, preds)
    
# Load the serialized face detector model from disk
prototxtPath = "face_detector/deploy.prototxt.txt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# Initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# Loop over the frames from the video stream
while True:
    # Grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    
    # Apply Gaussian Blur
    frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Apply Median Blur
    frame_blur = cv2.medianBlur(frame_blur, 5)
    
    # Apply Bilateral Filter
    frame_blur = cv2.bilateralFilter(frame_blur, 9, 75, 75)
    
    # Apply Non-Local Means Denoising
    frame_blur = cv2.fastNlMeansDenoisingColored(frame_blur, None, 10, 10, 7, 21)
    
    # Detect faces in the frame and determine if they are wearing a face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    
    # Loop over the detected face locations and their corresponding predictions
    for (box, pred) in zip(locs, preds):
        # Unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask, improperMask) = pred
        
        # Determine the class label and color we'll use to draw the bounding box and text
        if max(mask, withoutMask, improperMask) == mask:
            label = "Improper Mask"
            color = (255, 255, 0)
        elif min(mask, withoutMask, improperMask) == withoutMask:
            label = "No Mask"
            color = (0, 0, 255)
        else:
            label = "Mask"
            color = (0, 225, 0)
    
        # Include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask, improperMask) * 100)
    
        # Display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # If the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break
        
cv2.destroyAllWindows()
vs.stop()

