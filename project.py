import numpy as np
import cv2
import matplotlib.patches as patches


capL = cv2.VideoCapture('robotL.avi')
capR = cv2.VideoCapture('robotR.avi')
focal_lenght = 567.2  # in pixel
baseline = 92.226  # in mm
dim = 50

if not capL.isOpened():
    print("Error opening video stream or file")

if not capR.isOpened():
    print("Error opening video stream or file")

while capL.isOpened() or capR.isOpened():
    retL, frame = capL.read()
    retR, frame1 = capR.read()

    if retL is True and retR is True:
        # Capture frame-by-frame
        # Display the resulting frame
        cv2.rectangle(frame, (int(frame.shape[1]/2)-dim, int(frame.shape[0]/2)-dim), (int(frame.shape[1]/2)+dim, int(frame.shape[0]/2)+dim), 250, 1)
        cv2.rectangle(frame1, (int(frame1.shape[1]/2)-dim, int(frame1.shape[0]/2)-dim), (int(frame1.shape[1]/2)+dim, int(frame1.shape[0]/2)+dim), 250, 1)
        centerL = frame[int(frame.shape[0]/2)-dim: int(frame.shape[0]/2)+dim, int(frame.shape[1]/2)-dim:int(frame.shape[1]/2)+dim]
        centerR = frame1[int(frame1.shape[0] / 2) - dim: int(frame1.shape[0] / 2) + dim,
                  int(frame1.shape[1] / 2) - dim:int(frame1.shape[1] / 2) + dim]
        cv2.imshow('centerL',centerL)
        cv2.imshow('centerR', centerR)


        gray = cv2.cvtColor(centerL, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
    # draw key points detected
        img = cv2.drawKeypoints(gray, kp, gray, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("grayframe", img)

        gray1 = cv2.cvtColor(centerR, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        # draw key points detected
        img1 = cv2.drawKeypoints(gray1, kp1, gray1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("grayframe1", img1)


        cv2.imshow('FrameL', frame)
        cv2.imshow('FrameR', frame1)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
print('ciao')

capL.release()
capR.release()

cv2.destroyAllWindows()
