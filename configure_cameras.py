from cv2 import cv2

cam0 = cv2.VideoCapture(0)
cam1 = cv2.VideoCapture(1)
cam2 = cv2.VideoCapture(2)
while True:
    ret, frame = cam0.read()
    cv2.imshow('CAM0', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cam0.release()

while True:
    ret, frame = cam1.read()
    cv2.imshow('CAM1', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cam1.release()

while True:
    ret, frame = cam2.read()
    cv2.imshow('CAM2', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cam2.release()

cv2.destroyAllWindows()
