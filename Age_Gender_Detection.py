# This file is meant for show-casing the age and gender recognition abilities of the machine.

# People tracker and counter model: https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/
# Age and gender detection model: https://github.com/serengil/tensorflow-101/blob/master/python/age-gender-prediction-real-time.py
# Mask model: https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from cv2 import cv2
import numpy as np
import time
import os
import imutils
import dlib
import pika
import base64
import requests
import Age_Gender_Training
from Centroid_Tracker import Centroid_Tracker
from Trackable_Object import Trackable_Object

HAAR_FACE_DETECTOR = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
faceWeights = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceModel = "face_detector/deploy.prototxt"
maskModel = "mask_detector/mask_detector.model"
entryProto = "entry_detector/MobileNetSSD_deploy.prototxt"
entryModel = "entry_detector/MobileNetSSD_deploy.caffemodel"

ageList = np.array([i for i in range(0, 101)])
genderList = ['Male', 'Female']
classesList = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

faceNet = cv2.dnn.readNet(faceModel, faceWeights)
ageNet = Age_Gender_Training.age_model()
genderNet = Age_Gender_Training.gender_model()
maskNet = load_model(maskModel)
entryNet = cv2.dnn.readNetFromCaffe(entryProto, entryModel)
SKIP_FRAMES = 27
SKIP_FRAMES_COUNTER = 60
SERVER_IP = 'localhost'
prevOutFrames = []
prevInFrames = []
ct = Centroid_Tracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}
totalFrames = 0
entered = 0
exited = 0
totalIn = 0
totalOut = 0
H = None
W = None

def highlightFace(net, frame, conf_threshold=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    faces = []
    index = 1
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3]*w)
            y1 = int(detections[0, 0, i, 4]*h)
            x2 = int(detections[0, 0, i, 5]*w)
            y2 = int(detections[0, 0, i, 6]*h)
            faceBoxes.append([x1, y1, x2, y2])
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            faceWidth = endX - startX
            faceHeight = endY - startY
            # Save image of detected face as a square
            if faceHeight > faceWidth:
                reminder = int((faceHeight - faceWidth) / 2)
                startX -= reminder
                endX += reminder
            elif faceWidth > faceHeight:
                reminder = int((faceWidth - faceHeight) / 2)
                startY -= reminder
                endY += reminder

            face = frame[startY:endY, startX:endX]
            faceName = "./mask_detector/no_mask_faces/face" + str(index) + ".jpeg"
            cv2.imwrite(faceName, face)
            index += 1
            # Convert it from BGR to RGB channel ordering,
            # resize it to 224x224, and preprocess it
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            # Add the face and bounding boxes to their respective lists
            faces.append(face)
    return frame, faceBoxes, faces

def findPassingCustomers(frames, numPass):
    ages = []
    genders = []
    for frame in frames:
        faces = HAAR_FACE_DETECTOR.detectMultiScale(frame, scaleFactor=1.3,
        minNeighbors=5)
        # No faces were detected
        if len(faces) == 0:
            continue
        
        # More faces identifies in frame
        if len(faces) != numPass:
            new_faces = []
            # Find closest faces to camera
            for _ in range(numPass):
                if len(faces) == 0:
                    break
                # Find closest face according to its height
                face = max(faces, key=lambda item:item[3])
                new_faces.append(face)
                faces = np.delete(faces, np.argwhere(faces == face))
            faces = new_faces
        
        # Detect age and gender of faces
        for (x, y, w, h) in faces:
            # if w > 130:
            # Extract detected face
            detected_face = frame[int(y):int(y+h), int(x):int(x+w)]
            try:
                # Age and gender data set has 40% margin around the face - expand detected face
                margin = 30
                margin_x = int((w * margin)/100); margin_y = int((h * margin)/100)
                detected_face = frame[int(y-margin_y):int(y+h+margin_y),
                int(x-margin_x):int(x+w+margin_x)]
            except:
                print("detected face has no margin")
            try:
                # Vgg-face expects inputs (224, 224, 3)
                detected_face = cv2.resize(detected_face, (224, 224))
                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                img_pixels /= 255
                
                # Find out age
                age_distributions = ageNet.predict(img_pixels)
                age = str(int(np.floor(np.sum(age_distributions * ageList, axis = 1))[0]))
                ages.append(age)
                # Find out gender
                gender_distribution = genderNet.predict(img_pixels)[0]
                gender_index = np.argmax(gender_distribution)
                if gender_index == 0: gender = "F"
                else: gender = "M"
                genders.append(gender)

                return ages, genders

            except Exception as e:
                print("exception",str(e))
    
    return ages, genders

camFrontOut = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# camFrontIn = cv2.VideoCapture(3, cv2.CAP_DSHOW)
# camEntrance = cv2.VideoCapture(1, cv2.CAP_DSHOW)
time.sleep(2.0)
# connection = pika.BlockingConnection(pika.ConnectionParameters(SERVER_IP))
# channelExit = connection.channel()
# channelExit.queue_declare(queue='exited', durable=True)
# channelEnter = connection.channel()
# channelEnter.queue_declare(queue='entered', durable=True)
while True:
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Frontal Camera towards Outside
    # hasFrontOutFrame, frontOutFrame = camFrontOut.read()
    # if hasFrontOutFrame:
    #     # Detect every one second
    #     if totalFrames % SKIP_FRAMES == 0:
    #         # Save frame for later use
    #         prevOutFrames.append(frontOutFrame)
    #         # Delete existing faces with no masks
    #         noMaskFacesDir = "./mask_detector/no_mask_faces/"
    #         facesImg = os.listdir(noMaskFacesDir)
    #         for item in facesImg:
    #             if item.endswith(".jpeg"):
    #                 os.remove(os.path.join(noMaskFacesDir, item))
    #         # Find faces for mask detection
    #         resultImg, faceBoxes, faces = highlightFace(faceNet, frontOutFrame)
    #         if faceBoxes and faces:
    #             # Detect if people in line wearing masks or not
    #             faces = np.array(faces, dtype="float32")
    #             maskPreds = maskNet.predict(faces, batch_size=32)
    #             noMaskFaces = []
    #             index = 1
    #             for maskPred, faceBox in zip(maskPreds, faceBoxes):
    #                 (mask, withoutMask) = maskPred
    #                 mask = "Mask" if mask > withoutMask else "No Mask"
    #                 # Identifies person with no mask
    #                 if mask == "No Mask":
    #                     # Find saved image face with no mask and add to list
    #                     faceImgPath = noMaskFacesDir + "face" + str(index) + ".jpeg"
    #                     with open(faceImgPath, "rb") as img_file:
    #                         faceImg = base64.b64encode(img_file.read())
    #                         faceImgStr = 'data:image/jpeg;base64,' + faceImg.decode('utf-8')
    #                     noMaskFaces.append(faceImgStr)
    #                 index += 1
    #             # Send faces with no masks and number of people in line as json file to server
    #             entryStatus = {'images': noMaskFaces, 'waiting': len(faces)}
    #             r = requests.post('http://'+ SERVER_IP +':3000/entryStatus', json=entryStatus)
    #         else:
    #             entryStatus = {'images': [], 'waiting': 0}
    #             r = requests.post('http://'+ SERVER_IP +':3000/entryStatus', json=entryStatus)
    
    width = 3000
    hasFrontOutFrame, frontOutFrame = camFrontOut.read()
    if hasFrontOutFrame:
        frontOutFrame = imutils.resize(frontOutFrame, width=width)
        # Save frame for later use
        prevOutFrames.append(frontOutFrame)
        # Detect every one second
        if totalFrames % SKIP_FRAMES == 0:
            ages, genders = findPassingCustomers(prevOutFrames, 1)
            print(ages)
            print(genders)
            if len(ages) > 0 and len(genders) > 0:
                prevOutFrames.clear()
    # Frontal Camera towards Inside
    # hasFrontInFrame, frontInFrame = camFrontIn.read()
    # if hasFrontInFrame:
    #     frontInFrame = imutils.resize(frontInFrame, width=width)
    #     # Save frame for later use
    #     prevInFrames.append(frontInFrame)
    #     # Detect every one second
    #     if totalFrames % SKIP_FRAMES == 0:
    #         ages, genders = findPassingCustomers(prevInFrames, 2)
    #         print("Inside:")
    #         print(ages)
    #         print(genders)
    #         if len(ages) > 0 and len(genders) > 0:
    #             prevInFrames.clear()
    
    # Entrance/Exit Camera
    # hasEnteranceFrame, enteranceFrame = camEntrance.read()
    # if hasEnteranceFrame:
    #     enteranceFrame = imutils.resize(enteranceFrame, width=500)
    #     rgb = cv2.cvtColor(enteranceFrame, cv2.COLOR_BGR2RGB)
    #     # If frame dimensions are empty, set them
    #     if W is None or H is None:
    #         (H, W) = enteranceFrame.shape[:2]
    #     rects = []
    #     # Detect every one second
    #     if totalFrames % SKIP_FRAMES_COUNTER == 0:
    #         trackers = []
    #         blob = cv2.dnn.blobFromImage(enteranceFrame, 0.007843, (W, H), 127.5)
    #         entryNet.setInput(blob)
    #         detections = entryNet.forward()

    #         for i in np.arange(0, detections.shape[2]):
    #             # Extract the confidence (i.e., probability) associated with the prediction
    #             confidence = detections[0, 0, i, 2]

    #             # Filter out weak detections by requiring a minimum confidence
    #             if confidence > 0.4:
    #                 # Extract the index of the class label from the detections list
    #                 idx = int(detections[0, 0, i, 1])

    #                 # If the class label is not a person, ignore it
    #                 if classesList[idx] != "person":
    #                     continue

    #                 # Compute the (x, y)-coordinates of the bounding box for the object
    #                 box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
    #                 (startX, startY, endX, endY) = box.astype("int")

    #                 # Construct a dlib rectangle object from the bounding
    #                 # box coordinates and then start the dlib correlation tracker
    #                 tracker = dlib.correlation_tracker()
    #                 rect = dlib.rectangle(startX, startY, endX, endY)
    #                 tracker.start_track(rgb, rect)
    #                 trackers.append(tracker)
        
    #     # Otherwise, we should utilize our trackers rather than
    #     # detectors to obtain a higher frame processing throughput
    #     else:
    #         # hasEntered = False
    #         # Loop over the trackers
    #         for tracker in trackers:
    #             # Update the tracker and grab the updated position
    #             tracker.update(rgb)
    #             pos = tracker.get_position()
    #             # Unpack the position object
    #             startX = int(pos.left())
    #             startY = int(pos.top())
    #             endX = int(pos.right())
    #             endY = int(pos.bottom())
    #             # Add the bounding box coordinates to the rectangles list
    #             rects.append((startX, startY, endX, endY))
        
    #     # Draw a vertical line in the center of the frame - once an
    #     # object crosses this line we will determine whether they were
    #     # moving in or out of the store
    #     cv2.line(enteranceFrame, (W // 2, 0), (W // 2, H), (0, 255, 255), 2)
    #     objects = ct.update(rects)
    #     # Loop over the tracked objects
    #     for (objectID, centroid) in objects.items():
    #         # Check if a trackable object exists for the current object ID
    #         to = trackableObjects.get(objectID, None)
    #         # If there is no existing trackable object, create one
    #         if to is None:
    #             to = Trackable_Object(objectID, centroid)

    #         # There is a trackable object to utilize to determine direction
    #         else:
    #             entered = 0
    #             exited = 0
    #             # The difference between the x-coordinate of the current
    #             # centroid and the mean of previous centroids will tell
    #             # us in which direction the object is moving (negative for
    #             # 'left' and positive for 'right')
    #             x = [c[0] for c in to.centroids]
    #             direction = centroid[0] - np.mean(x)
    #             to.centroids.append(centroid)

    #             # Check to see if the object has been counted or not
    #             if not to.counted:
    #                 # If the direction is negative (indicating the object
    #                 # is moving to the left) AND the centroid is to the left
    #                 # of the center line, count the object as exiting
    #                 if direction < 0 and centroid[0] < W // 2:
    #                     totalIn += 1
    #                     entered += 1
    #                     print('Person entered')
    #                     print(entered)
    #                     to.counted = True

    #                 # If the direction is positive (indicating the object
    #                 # is moving to the right) AND the centroid is to the
    #                 # right of the center line, count the object as entering
    #                 elif direction > 0 and centroid[0] > W // 2:
    #                     exited += 1
    #                     totalOut += 1
    #                     print('Person exited')
    #                     print(exited)
    #                     to.counted = True

    #         # Store the trackable object in dictionary
    #         trackableObjects[objectID] = to

    #     # People have entered the store
    #     if entered > 0:
    #         ages = []
    #         genders = []
    #         # Detect the age and gender of the customers that entered
    #         ages, genders = findPassingCustomers(prevOutFrames, entered)
    #         if len(ages) == 0 and len(genders) == 0:
    #             message = "M-0"
    #             print(message)
    #             channelEnter.basic_publish(exchange='', routing_key='entered', body=message)
    #         else:
    #             for age, gender in zip(ages, genders):
    #                 message = gender + "-" + str(age)
    #                 print(message)
    #                 channelEnter.basic_publish(exchange='', routing_key='entered', body=message)
        
    #     # People have left the store
    #     if exited > 0:
    #         ages = []
    #         genders = []
    #         # Detect the age and gender of the customers that left
    #         ages, genders = findPassingCustomers(prevInFrames, exited)
    #         if len(ages) == 0 and len(genders) == 0:
    #             message = "M-0"
    #             print(message)
    #             channelExit.basic_publish(exchange='', routing_key='exited', body=message)
    #         else:
    #             for age, gender in zip(ages, genders):
    #                 message = gender + "-" + str(age)
    #                 print(message)
    #                 channelExit.basic_publish(exchange='', routing_key='exited', body=message)
    
    # People have entered the store
    # if entered > 0:
    #     ages = []
    #     genders = []
    #     # Detect the age and gender of the customers that entered
    #     ages, genders = findPassingCustomers(prevOutFrames, entered)
    #     if len(ages) == 0 and len(genders) == 0:
    #         message = "M-0"
    #         print(message)
    #         channelEnter.basic_publish(exchange='', routing_key='entered', body=message)
    #     else:
    #         for age, gender in zip(ages, genders):
    #             message = gender + "-" + str(age)
    #             print(message)
    #             channelEnter.basic_publish(exchange='', routing_key='entered', body=message)
        
    # # People have left the store
    # if exited > 0:
    #     ages = []
    #     genders = []
    #     # Detect the age and gender of the customers that left
    #     ages, genders = findPassingCustomers(prevInFrames, exited)
    #     if len(ages) == 0 and len(genders) == 0:
    #         message = "M-0"
    #         print(message)
    #         channelExit.basic_publish(exchange='', routing_key='exited', body=message)
    #     else:
    #         for age, gender in zip(ages, genders):
    #             message = gender + "-" + str(age)
    #             print(message)
    #             channelExit.basic_publish(exchange='', routing_key='exited', body=message)
    # Save only 100000 seconds of frames
    if len(prevOutFrames) > 30:
        prevOutFrames.clear()
    # if len(prevInFrames) > 30:
    #     prevInFrames.clear()
    
    totalFrames += 1

camFrontOut.release()
# camFrontIn.release()
# camEntrance.release()
# connection.close()
cv2.destroyAllWindows()
