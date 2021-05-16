from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from cv2 import cv2
import numpy as np
import imutils
import pika
import Age_Gender_Training
import Centroid_Tracker
import Trackable_Object


faceProto = "face_detector/opencv_face_detector.pbtxt"
faceModel = "face_detector/opencv_face_detector_uint8.pb"
maskModel = "mask_detector/mask_detector.model"
entryProto = "entry_detector/MobileNetSSD_deploy.prototxt"
entryModel = "entry_detector/MobileNetSSD_deploy.caffemodel"

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
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
totalDown = 0
totalUp = 0
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
        if totalFrames % SKIP_FRAMES == 0:
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
            # TODO: detect and track if entered the store - if yes, find gender and age
    
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
