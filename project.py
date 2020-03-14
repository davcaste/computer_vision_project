import numpy as np
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt


capL = cv2.VideoCapture('robotL.avi')
capR = cv2.VideoCapture('robotR.avi')
focal_lenght = 567.2  # in pixel
baseline = 92.226  # in mm
dim = 100

if not capL.isOpened():
    print("Error opening video stream or file")

if not capR.isOpened():
    print("Error opening video stream or file")

while capL.isOpened() and capR.isOpened():
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if retL is True and retR is True:
        # Capture frame-by-frame
        # Display the resulting frame
        cv2.rectangle(frameL, (int(frameL.shape[1]/2)-dim, int(frameL.shape[0]/2)-dim), (int(frameL.shape[1]/2)+dim, int(frameL.shape[0]/2)+dim), 250, 1)
        cv2.rectangle(frameR, (int(frameR.shape[1]/2)-dim, int(frameR.shape[0]/2)-dim), (int(frameR.shape[1]/2)+dim, int(frameR.shape[0]/2)+dim), 250, 1)
        centerL = frameL[int(frameL.shape[0]/2)-dim: int(frameL.shape[0]/2)+dim, int(frameL.shape[1]/2)-dim:int(frameL.shape[1]/2)+dim]
        centerR = frameR[int(frameR.shape[0] / 2) - dim: int(frameR.shape[0] / 2) + dim, int(frameR.shape[1] / 2) - dim:int(frameR.shape[1] / 2) + dim]

        grayL = cv2.cvtColor(centerL, cv2.COLOR_BGR2GRAY)
        intermediateL = cv2.equalizeHist(grayL)
        finalL = cv2.medianBlur(intermediateL, 5)

        sift = cv2.xfeatures2d.SIFT_create()
        kp2, des2 = sift.detectAndCompute(finalL, None)
    # draw key points detected
        img2 = cv2.drawKeypoints(finalL, kp2, finalL, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("grayframe", img2)

        grayR = cv2.cvtColor(centerR, cv2.COLOR_BGR2GRAY)
        intermediateR = cv2.equalizeHist(grayR)
        finalR = cv2.medianBlur(intermediateR, 5)

        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(finalR, None)


        # draw key points detected
        img1 = cv2.drawKeypoints(finalR, kp1, finalR, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("grayframe1", img1)

        distance = np.array(np.sqrt(np.sum((des1[:, np.newaxis, :] - des2[np.newaxis, :, :]) ** 2, axis=-1))) #SIFT descriptor of a point is just 128-dimensional vector, so you can simple compute Euclidean distance between every two and match nearest pairs.

        # for k in range(len(kp1)):
        #     for z in range(len(kp2)):
        #         if kp1[k].angle - kp2[z].angle > 0.5 or kp1[k].angle - kp2[z].angle < -0.5:
        #             distance[k, z] = 100000000000
        #         if kp1[k].pt[1] - kp2[z].pt[1] > 3 or kp1[k].pt[1] - kp2[z].pt[1] < -3:
        #             distance[k, z] = 100000000000

        ind = np.unravel_index(np.argmin(distance, axis=None), distance.shape)
        print(distance.min())
        kpL = kp2[ind[1]]
        kpR = kp1[ind[0]]
        cv2.circle(centerL, (int(kpL.pt[0]), int(kpL.pt[1])), 4, (255, 0, 0), 1)
        cv2.imshow('centerL', centerL)
        cv2.circle(centerR, (int(kpR.pt[0]), int(kpR.pt[1])), 4, (255, 0, 0), 1)
        cv2.imshow('centerR', centerR)
        # # create BFMatcher object
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        #
        # # Match descriptors.
        # matches = bf.match(des1, des2)
        #
        # # Sort them in the order of their distance.
        # matches = sorted(matches, key=lambda x: x.distance)
        #
        # # Draw first 10 matches.
        # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], flags=2)
        #
        # plt.imshow(img3), plt.show()

        cv2.imshow('FrameL', frameL)
        cv2.imshow('FrameR', frameR)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

capL.release()
capR.release()

cv2.destroyAllWindows()
# print(distance)
print('ciao')
print(distance.min())
ind = np.unravel_index(np.argmin(distance, axis=None), distance.shape)
# print(ind)
# print(distance[ind])
kpL = kp2[ind[1]]
kpR = kp1[ind[0]]
cv2.circle(centerL, (int(kpL.pt[0]),int(kpL.pt[1])), 4, (255, 0, 0), 1)
cv2.imshow('one keypoint', centerL)
cv2.circle(centerR, (int(kpR.pt[0]),int(kpR.pt[1])), 4, (255, 0, 0), 1)
cv2.imshow('other keypoint', centerR)

# gray = cv2.cvtColor(centerL, cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
#     # draw key points detected
# img2 = cv2.drawKeypoints(gray, kpL, gray, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow("grayframe", img2)
#
# gray = cv2.cvtColor(centerL, cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
#     # draw key points detected
# img2 = cv2.drawKeypoints(gray, kpR, gray, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow("grayframe", img2)

#(row_index, col_index) = distance.argmin(0), distance.argmin(1)
# ri = distance[row_index, np.arange(len(row_index))]
# ci = distance[col_index, np.arange(len(col_index))]
# print(ri)
print('ciao')
# print(ci)
#print(distance[row_index,col_index])
# print(des1)
# print('ciao')
# print(des2)

