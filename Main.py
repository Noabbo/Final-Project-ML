# People tracker and counter model: https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/
# Age and gender detection model: https://github.com/serengil/tensorflow-101/blob/master/python/age-gender-prediction-real-time.py
# Mask model: https://github.com/J-Douglas/Face-Mask-Detection

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from cv2 import cv2
import numpy as np
import imutils
import dlib
import pika
import Age_Gender_Training
import Centroid_Tracker
import Trackable_Object

HAAR_FACE_DETECTOR = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
faceProto = "face_detector/opencv_face_detector.pbtxt"
faceModel = "face_detector/opencv_face_detector_uint8.pb"
maskModel = "mask_detector/mask_detector.model"
entryProto = "entry_detector/MobileNetSSD_deploy.prototxt"
entryModel = "entry_detector/MobileNetSSD_deploy.caffemodel"

ageList = np.array([i for i in range(0, 101)])
genderList = ['Male', 'Female']
classesList = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = Age_Gender_Training.age_model()
genderNet = Age_Gender_Training.gender_model()
maskNet = load_model(maskModel)
entryNet = cv2.dnn.readNetFromCaffe(entryProto, entryModel)
SKIP_FRAMES = 30
prevOutFrames = []
prevInFrames = []
hasEntered = False
ct = Centroid_Tracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}
totalFrames = 0
totalIn = 0
totalOut = 0
H = None
W = None

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3]*frameWidth)
            y1 = int(detections[0, 0, i, 4]*frameHeight)
            x2 = int(detections[0, 0, i, 5]*frameWidth)
            y2 = int(detections[0, 0, i, 6]*frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
            (startX, startY, endX, endY) = box.astype("int")
            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(frameWidth - 1, endX), min(frameHeight - 1, endY))
            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            # add the face and bounding boxes to their respective lists
            faces.append(face)
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes, faces

def findPassingCustomers(frames, numPass):
    ages = []
    genders = []
    for frame in frames:
        faces = HAAR_FACE_DETECTOR.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
        # No faces were detected
        if len(faces) == 0:
            continue
        
        # More faces identifies in frame
        if len(faces) != numPass:
            new_faces = []
            # Find closest faces to camera
            for _ in range(numPass):
                # Find closest face according to its height
                face = max(faces,key=lambda item:item[3])
                new_faces.append(face)
                faces.remove(face)
            faces = new_faces
        
        # Detect age and gender of faces
        for (x, y, w, h) in faces:
                if w > 130:
                    #extract detected face
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
                        #vgg-face expects inputs (224, 224, 3)
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
                        if gender_index == 0: gender = "Female"
                        else: gender = "Male"
                        genders.append(gender)

                    except Exception as e:
                        print("exception",str(e))
        break
    
    return ages, genders

camFrontOut = cv2.VideoCapture(0)
camFrontIn = cv2.VideoCapture(1)
camEntrance = cv2.VideoCapture(2)
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channelExit = connection.channel()
channelExit.queue_declare(queue='exited')
channelEnter = connection.channel()
channelEnter.queue_declare(queue='entered')
while True:
    # Frontal Camera towards Outside
    hasFrontOutFrame, frontOutFrame = camFrontOut.read()
    if hasFrontOutFrame:
        # Detect every one second
        if totalFrames % (SKIP_FRAMES * 5) == 0:
            # Save frame for later use
            prevOutFrames.append(frontOutFrame)
            # Find faces for mask detection
            resultImg, faceBoxes, faces = highlightFace(faceNet, frontOutFrame)
            if faceBoxes and faces:
                # Detect if people in line wearing masks or not
                faces = np.array(faces, dtype="float32")
                maskPreds = maskNet.predict(faces, batch_size=32)
                facesNoMask = []
                for maskPred, faceBox in zip(maskPreds, faceBoxes):
                    (mask, withoutMask) = maskPred
                    mask = "Mask" if mask > withoutMask else "No Mask"
                    # cv2.putText(resultImg, f'{mask}', (faceBox[0], faceBox[1]-10),
                    # cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                    # cv2.imshow("LIVE", resultImg)
                    # Identifies person with no mask
                    if mask == "No Mask":
                        facesNoMask.append(faceBox)
                # TODO: send faces with no mask to server
    
    # Frontal Camera towards Inside
    hasFrontInFrame, frontInFrame = camFrontIn.read()
    if hasFrontInFrame:
        # Detect every one second
        if totalFrames % SKIP_FRAMES == 0:
            # Save frame for later use
            prevOutFrames.append(frontInFrame)
    
    # Entrance/Exit Camera
    hasEnteranceFrame, enteranceFrame = camEntrance.read()
    if hasEnteranceFrame:
        enteranceFrame = imutils.resize(enteranceFrame, width=500)
        rgb = cv2.cvtColor(enteranceFrame, cv2.COLOR_BGR2RGB)
        rects = []
        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = enteranceFrame.shape[:2]
        # Detect every one second
        if totalFrames % SKIP_FRAMES == 0:
            trackers = []
            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections
            blob = cv2.dnn.blobFromImage(enteranceFrame, 0.007843, (W, H), 127.5)
            entryNet.setInput(blob)
            detections = entryNet.forward()
            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by requiring a minimum
                # confidence
                if confidence > 0.4:
                    # extract the index of the class label from the
                    # detections list
                    idx = int(detections[0, 0, i, 1])

                    # if the class label is not a person, ignore it
                    if classesList[idx] != "person":
                        continue

                    # compute the (x, y)-coordinates of the bounding box for the object
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                    tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    trackers.append(tracker)
        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            hasEntered = False
            # loop over the trackers
            for tracker in trackers:

                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))
        
        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'
        cv2.line(enteranceFrame, (W // 2, 0), (W // 2, H), (0, 255, 255), 2)

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = Trackable_Object(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                entered = 0
                exited = 0
                # the difference between the x-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'left' and positive for 'right')
                x = [c[0] for c in to.centroids]
                direction = centroid[0] - np.mean(x)
                to.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving to the left) AND the centroid is above the center
                    # line, count the object
                    if direction < 0 and centroid[1] < W // 2:
                        entered += 1
                        totalIn += 1
                        to.counted = True

                    # if the direction is positive (indicating the object
                    # is moving to the right) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] > W // 2:
                        totalOut += 1
                        exited += 1
                        to.counted = True

            # store the trackable object in dictionary
            trackableObjects[objectID] = to

        # People have entered the store
        if entered > 0:
            # Detect the age and gender of the customers that entered
            ages, genders = findPassingCustomers(prevInFrames, entered)
            # TODO: send via message queue the details
        
        # People have left the store
        if exited > 0:
            # Detect the age and gender of the customers that left
            ages, genders = findPassingCustomers(prevInFrames, exited)
            # TODO: send via message queue the details
    
    # Save only 10 seconds of frames
    if len(prevOutFrames) > 10:
        prevOutFrames.clear()
    if len(prevInFrames) > 10:
        prevInFrames.clear()
    
    totalFrames += 1

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

camFrontOut.release()
camFrontIn.release()
camEntrance.release()
connection.close()
cv2.destroyAllWindows()
