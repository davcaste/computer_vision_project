import numpy as np
import cv2

capL = cv2.VideoCapture('robotL.avi')
capR = cv2.VideoCapture('robotR.avi')

if not capL.isOpened():
    print("Error opening video stream or file")

if not capR.isOpened():
    print("Error opening video stream or file")

while (True):
    retL, frame = capL.read()
    retR, frame1 = capR.read()
    if retL is True and retR is True:
        # Capture frame-by-frame

        # Display the resulting frame
        cv2.imshow('FrameL', frame)
        cv2.imshow('FrameR', frame1)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


capL.release()
capR.release()

cv2.destroyAllWindows()