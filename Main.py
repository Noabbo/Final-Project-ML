from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from cv2 import cv2
import numpy as np
import argparse
import pika
import Age_Gender_Training


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

parser = argparse.ArgumentParser()
parser.add_argument('--image')
args = parser.parse_args()

faceProto = "face_detector/opencv_face_detector.pbtxt"
faceModel = "face_detector/opencv_face_detector_uint8.pb"
maskModel = "mask_detector/mask_detector.model"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = Age_Gender_Training.age_model()
genderNet = Age_Gender_Training.gender_model()
maskNet = load_model(maskModel)

prevOutFrames = []
prevInFrames = []
camFrontOut = cv2.VideoCapture(0)
camFrontIn = cv2.VideoCapture(1)
camEntrance = cv2.VideoCapture(2)
isNoMask = True
hasEntered = False
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channelOut = connection.channel()
channelOut.queue_declare(queue='exited')
channelIn = connection.channel()
channelIn.queue_declare(queue='entered')
while True:
    hasFrontOutFrame, frontOutFrame = camFrontOut.read()
    if hasFrontOutFrame:
        resultImg, faceBoxes, faces = highlightFace(faceNet, frontOutFrame)
        if not faceBoxes:
            print("No face detected from outside")
            continue

        # Save frame for later use
        prevOutFrames.append(frontOutFrame)
        # Detect if people in line wearing masks or not
        faces = np.array(faces, dtype="float32")
        maskPreds = maskNet.predict(faces, batch_size=32)
        isAllMask = True
        for maskPred, faceBox in zip(maskPreds, faceBoxes):
            (mask, withoutMask) = maskPred
            mask = "Mask" if mask > withoutMask else "No Mask"
            # cv2.putText(resultImg, f'{mask}', (faceBox[0], faceBox[1]-10),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
            # cv2.imshow("LIVE", resultImg)
            # Identifies person with no mask
            if mask == "No Mask" and isNoMask:
                isNoMask = False
                isAllMask = False
                break
        # All people are with masks - corrected the situation
        if isAllMask and not isNoMask:
            isAllMask = True
    
    hasFrontInFrame, frontInFrame = camFrontIn.read()
    if hasFrontInFrame:
        resultImg, faceBoxes, faces = highlightFace(faceNet, frontInFrame)
        if not faceBoxes:
            print("No face detected from inside")
            continue
        # Save frame for later use
        prevOutFrames.append(frontInFrame)
    
    # TODO: detect and track if entered the store - if yes, find gender and age
    
    # Save only five seconds of frames
    if len(prevOutFrames) > (28 * 5):
        prevOutFrames.clear()
    if len(prevInFrames) > (28 * 5):
        prevInFrames.clear()

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

camFrontOut.release()
connection.close()
cv2.destroyAllWindows()
