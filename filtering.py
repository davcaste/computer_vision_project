import numpy as np
import cv2
from skimage import filters
import matplotlib.pyplot as plt
from scipy.signal import lfilter

capL = cv2.VideoCapture('robotL.avi')
capR = cv2.VideoCapture('robotR.avi')
focal_lenght = 567.2  # in pixel
baseline = 92.226  # in mm
dim_kp = 30
h_stripe = 2
max_kp = []
dist = 3
dist_vect = np.ones((5,), dtype='int') * dim_kp
d_main = ((baseline * focal_lenght) / dist) * 0.001
d_main_object = d_main
d_main_background = d_main
dist_ob = dist
dist_bg = dist
temp_dim = dim_kp
flag = False
min_disp = 0
max_disp = 64
tot_dist = []
tot_dist_bg = []
count_o = 0
count1 = 0
count_b = 0
filtered_dist_obj = [d_main, d_main]
filtered_dist_bkg = [d_main, d_main]
first_o = True
first_b = True
treshold_o = 0.5

n_frame = 0
tot_dist_filt = []
tot_dist_bg_filt = []


#------------------------------------------------
# Create a FIR filter and apply it to x.
#------------------------------------------------

# Define cutoff frequency
frame_rate = 15
Cutoff_freq = 0.5
d0 = 3

# Calculate Nyquist frequency
Nyq_frequency = frame_rate / 2
Cutoff_norm = Cutoff_freq / Nyq_frequency
# FIR order
order = 6
# Coefficinets of the FIR filter
FIR_coeff = [-0.00545737100067199, 0.0317209689414059, 0.254972364809816, 0.437528074498901, 0.254972364809816, 0.0317209689414059, -0.00545737100067199]
WS = len(FIR_coeff)
x = list(3*np.ones(WS))
y = list(3*np.ones(WS))
x_bg = list(3*np.ones(WS))
y_bg = list(3*np.ones(WS))

def image_initialization(frame_L, frameR, dim_kp):
    # Capture frame-by-frame
    # Display the resulting frame
    cv2.rectangle(frameL, (int(frameL.shape[1] / 2) - dim_kp, int(frameL.shape[0] / 2) - dim_kp),
                  (int(frameL.shape[1] / 2) + dim_kp, int(frameL.shape[0] / 2) + dim_kp), 250, 1)
    cv2.rectangle(frameR, (int(frameR.shape[1] / 2) - dim_kp, int(frameR.shape[0] / 2) - dim_kp),
                  (int(frameR.shape[1] / 2) + dim_kp, int(frameR.shape[0] / 2) + dim_kp), 250, 1)
    centerL = frameL[int(frameL.shape[0] / 2) - dim_kp: int(frameL.shape[0] / 2) + dim_kp,
              int(frameL.shape[1] / 2) - dim_kp:int(frameL.shape[1] / 2) + dim_kp]
    centerR = frameR[int(frameR.shape[0] / 2) - dim_kp: int(frameR.shape[0] / 2) + dim_kp,
              int(frameR.shape[1] / 2) - dim_kp:int(frameR.shape[1] / 2) + dim_kp]
    total_grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    total_intermediateL = cv2.equalizeHist(total_grayL)
    total_finalL = total_intermediateL
    grayL = cv2.cvtColor(centerL, cv2.COLOR_BGR2GRAY)
    intermediateL = cv2.equalizeHist(grayL)
    finalL = cv2.medianBlur(intermediateL, 5)
    # cv2.bilateralFilter(grayL, 5, 3, 3)

    # cv2.equalizeHist(intermediateL)

    # intermediateL

    # cv2.medianBlur(intermediateL, 5)

    # cv2.GaussianBlur(intermediateL, (3, 3), 3)
    grayR = cv2.cvtColor(centerR, cv2.COLOR_BGR2GRAY)
    intermediateR = cv2.equalizeHist(grayR)
    finalR = cv2.medianBlur(intermediateR, 5)
    # cv2.bilateralFilter(grayR, 5, 3, 3)

    # cv2.equalizeHist(intermediateR)

    # intermediateR

    # cv2.medianBlur(intermediateR, 5)
    return finalL, finalR

def keypoints_division(kpL, kpR, desL, desR, h_stripe, n_stripes):
    # image_stripesL = np.zeros((int(finalL.shape[1] / h_stripe), h_stripe, finalL.shape[0]),
    #                           dtype='uint8')  # NxHxL where N = n of stripes, H = height of stripes, L = lenght of window
    # image_stripesR = np.zeros((int(finalR.shape[1] / h_stripe), h_stripe, finalR.shape[0]), dtype='uint8')
    kp_stripesL = []
    kp_stripesR = []
    des_stripesL = []
    des_stripesR = []

    # cv2.imshow('f',frameL)
    # FLANN parameters
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    # matches = np.zeros((int(finalL.shape[1] / h_stripe)), dtype='object')
    # good = []
    # ptsL = []
    # ptsR = []
    #
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # for i in range(int(finalL.shape[1] / h_stripe)):
    #     if kp_stripesL[i] == 0:
    #         kp_stripesL[i] = []
    #     if kp_stripesR[i] == 0:
    #         kp_stripesR[i] = []
    #     for j in range(len(kp2)):
    #         if i*h_stripe <= kp2[j].pt[1] < (i+1)*h_stripe:
    #             kp_stripesL[i].append(kp2[j])
    #     for j in range(len(kp1)):
    #         if i * h_stripe <= kp1[j].pt[1] < (i + 1) * h_stripe:
    #             kp_stripesR[i].append(kp1[j])

    for s in range(n_stripes):
        kp_stripesL.append([])
        kp_stripesR.append([])
        des_stripesL.append([])
        des_stripesR.append([])
        for j in range(len(kpL)):
            if s * h_stripe <= kpL[j].pt[1] < (s + 1) * h_stripe:
                kp_stripesL[s].append(kpL[j])
                des_stripesL[s].append(desL[j])
        for j in range(len(kpR)):
            if s * h_stripe <= kpR[j].pt[1] < (s + 1) * h_stripe:
                kp_stripesR[s].append(kpR[j])
                des_stripesR[s].append(desR[j])
    # print(len(des_stripesL))
    des_stripesL = np.array(des_stripesL)
    kp_stripesL = np.array(kp_stripesL)
    des_stripesR = np.array(des_stripesR)
    kp_stripesR = np.array(kp_stripesR)
    return kp_stripesL, des_stripesL, kp_stripesR, des_stripesR

def estimation_chessboard(frameL):
    found, corners = cv2.findChessboardCorners(frameL, (6,8))
    cv2.drawChessboardCorners(frameL, (6,8), corners, found)
    if found:
        pt = (corners[0],corners[-1])
        cv2.circle(frameL,(pt[0][0][0],pt[0][0][1]),8,(0,255,0),1)
        cv2.circle(frameL,(pt[1][0][0],pt[1][0][1]),8,(0,255,0),1)
        l_chess = pt[1][0][0] - pt[0][0][0]
        h_chess = pt[1][0][1] - pt[0][0][1]
        l_chess_mm = ((dist_ob*l_chess)/focal_lenght) * 1000
        h_chess_mm = ((dist_ob*h_chess)/focal_lenght) *1000
        print('l, h in pixels', l_chess, h_chess)
        print('l, h in mm', l_chess_mm,h_chess_mm)


def disparity_map_calculation(kp_stripesL, des_stripesL, kp_stripesR, des_stripesR):
    distance = []
    disparity_map = []
    kp_struct = []
    # distance = np.zeros((int(finalR.shape[1] / h_stripe),), dtype='object')

    # distance[i, :, :] = np.array(np.sqrt(np.sum((des_stripesL[i][:, np.newaxis, :] - des_stripesR[i][np.newaxis, :, :]) ** 2, axis=-1)))

    # distance = np.array(np.sqrt(np.sum((des1[:, np.newaxis, :] - des2[np.newaxis, :, :]) ** 2, axis=-1)))

    for j in range(len(des_stripesL)):
        des_stripesL[j] = np.array(des_stripesL[j])
        kp_stripesL[j] = np.array(kp_stripesL[j])
        des_stripesR[j] = np.array(des_stripesR[j])
        kp_stripesR[j] = np.array(kp_stripesR[j])
        distance.append([])
        if len(kp_stripesL[j]) > 0 and len(kp_stripesR[j]) > 0:
            distance[j] = np.array(np.sqrt(
                np.sum((des_stripesL[j][:, np.newaxis, :] - des_stripesR[j][np.newaxis, :, :]) ** 2, axis=-1)))
            ind = tuple(zip(*np.where(distance[j] < 150)))
            index = list(ind)
            removed = np.ones(len(ind), dtype='uint8')
            for y in range(len(ind) - 1):
                check = False
                if removed[y] == 1:
                    for z in range(y + 1, len(ind)):
                        if ind[y][0] == ind[z][0] or ind[y][1] == ind[z][1]:
                            if distance[j][ind[y]] < distance[j][ind[z]]:
                                if removed[z] == 1:
                                    # print(ind[k])
                                    index.remove(ind[z])
                                    removed[z] = 0
                            else:
                                if removed[y] == 1:
                                    # print('ciao', ind[i])
                                    index.remove(ind[y])
                                    removed[y] = 0
            index = tuple(index)

            for y in index:
                uL = int(kp_stripesL[j][y[0]].pt[0] + off_x_L)
                uR = int(kp_stripesR[j][y[1]].pt[0] + off_x_R)
                vL = int(kp_stripesL[j][y[0]].pt[1] + off_y_L)
                vR = int(kp_stripesR[j][y[1]].pt[1] + off_y_R)
                disparity = uL - uR
                kp_struct.append(((uL, vL), (uR, vR)))
                if disparity + frameL.shape[1] > 0:
                    disparity_map.append(disparity + frameL.shape[1])
                    cv2.circle(outimg, (uL, vL), 3, (255, 0, 0), 1)
                    cv2.circle(outimg, (uR, vR), 3, (255, 0, 0), 1)
                    cv2.line(outimg, (uL, vL), (uR, vR), (255, 0, 0), 1)
    num_kp = len(kp_struct)
    return disparity_map, num_kp


if not capL.isOpened():
    print("Error opening video stream or file")

if not capR.isOpened():
    print("Error opening video stream or file")

sift = cv2.xfeatures2d.SIFT_create()

while capL.isOpened() and capR.isOpened():

    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if retL is True and retR is True:
        finalL, finalR = image_initialization(frameL, frameR, dim_kp)
        info_frame = np.empty_like(finalL)
        H = frameL.shape[0]
        L = frameL.shape[1]
        #
        h = finalL.shape[0]
        l = finalL.shape[1]
        #
        off_x_L = (L / 2 - l / 2)
        # int(L/2)
        off_y_L = (H / 2 - h / 2)
        # int(H/2)
        off_x_R = off_x_L + frameL.shape[1]
        off_y_R = off_y_L
        # #     STATE OF ART FOUND OF CV2

        # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

        # disparity = stereo.compute(finalL, finalR)

        # d = disparity/255

        # #disparity = np.uint8(disparity)

        # cv2.imshow('title',disparity/255)

        # cv2.imshow('t1',finalL)


        # center_tempL = frameL[int(frameL.shape[0] / 2) - dim: int(frameL.shape[0] / 2) + dim,

        #           int(frameL.shape[1] / 2) - dim:int(frameL.shape[1] / 2) + dim]

        # center_tempL = copy.deepcopy(centerL)

        # # center_tempR = frameR[int(frameR.shape[0] / 2) - dim: int(frameR.shape[0] / 2) + dim,

        # #           int(frameR.shape[1] / 2) - dim:int(frameR.shape[1] / 2) + dim]

        # center_tempR = copy.deepcopy(centerR)

        #

        # grayL = cv2.cvtColor(centerL, cv2.COLOR_BGR2GRAY)

        # intermediateL = cv2.equalizeHist(grayL)

        # finalL = cv2.medianBlur(intermediateL, 5)

        #

        # grayR = cv2.cvtColor(centerR, cv2.COLOR_BGR2GRAY)

        # intermediateR = cv2.equalizeHist(grayR)

        # finalR = cv2.medianBlur(intermediateR, 5)

        # #     DIVISION IN STRIPES BEFORE COMPUTING THE KEYPOINTS

        # image_stripesL = np.zeros((int(finalL.shape[1]/h_stripe), h_stripe, finalL.shape[0]), dtype='uint8')  # NxHxL where N = n of stripes, H = height of stripes, L = lenght of window

        # image_stripesR = np.zeros((int(finalR.shape[1] / h_stripe), h_stripe, finalR.shape[0]), dtype='uint8')

        # kpL = np.zeros((int(finalL.shape[1]/h_stripe)), dtype='object')

        # desL = np.zeros(int(finalL.shape[1]/h_stripe), dtype='object')

        # imgL = np.zeros((int(finalL.shape[1]/h_stripe), h_stripe, finalL.shape[0]), dtype='uint8')

        #

        # kpR = np.zeros((int(finalR.shape[1]/h_stripe)), dtype='object')

        # desR = np.zeros(int(finalR.shape[1]/h_stripe), dtype='object')

        # imgR = np.zeros((int(finalR.shape[1]/h_stripe), h_stripe, finalL.shape[0]), dtype='uint8')

        # distance = np.zeros((int(finalR.shape[1] / h_stripe), ), dtype='object')

        # maxL = 0

        # maxR = 0

        #

        # sift = cv2.xfeatures2d.SIFT_create()

        #

        # for i in range(int(finalL.shape[1]/h_stripe)):

        #     image_stripesL[i] = finalL[i*h_stripe:(i+1)*h_stripe, :]

        #     image_stripesR[i] = finalR[i * h_stripe:(i + 1) * h_stripe, :]

        #     kpL[i], desL[i] = sift.detectAndCompute(image_stripesL[i], None)

        #     kpR[i], desR[i] = sift.detectAndCompute(image_stripesR[i], None)

        #     try:

        #         distance[i] = np.zeros((desL[i].shape[0], desR[i].shape[0]))

        #         center_tempL[i*h_stripe:(i+1)*h_stripe,:,:] = cv2.drawKeypoints(image_stripesL[i], kpL[i], center_tempL[i], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #         center_tempR[i*h_stripe:(i+1)*h_stripe,:,:] = cv2.drawKeypoints(image_stripesR[i], kpR[i], center_tempR[i], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #         distance[i] = np.array(np.sqrt(np.sum((desL[i][:, np.newaxis, :] - desR[i][np.newaxis, :, :]) ** 2, axis=-1)))

        #     except(AttributeError):

        #         distance[i] = np.array([[1000000]])

        #     # if len(kpL[i]) > maxL:

        #     #     maxL = len(kpL[i])

        #     # if len(kpR[i]) > maxR:

        #     #     maxR = len(kpR[i])

        #

        # totalimageL = np.vstack([center_tempL[i*h_stripe:(i+1)*h_stripe,:,:] for i in range(int(finalL.shape[1]/h_stripe))])

        # totalimageR = np.vstack(

        #     [center_tempR[i * h_stripe:(i + 1) * h_stripe, :, :] for i in range(int(finalR.shape[1] / h_stripe))])

        # kptL = np.empty((sum([len([kpL[i] for i in range(int(finalL.shape[1]/h_stripe))])])), dtype='object')

        # kptR = np.empty((sum([len([kpR[i] for i in range(int(finalR.shape[1] / h_stripe))])])), dtype='object')

        # destL = []

        # destR = []

        # kptL = np.concatenate([kpL[i] for i in range(int(finalL.shape[1]/h_stripe))]).ravel()

        # kptR = np.concatenate([kpR[i] for i in range(int(finalR.shape[1] / h_stripe))]).ravel()

        #

        # destL.append([desL[i] for i in range(int(finalL.shape[1]/h_stripe))])

        # destR.append([desR[i] for i in range(int(finalR.shape[1] / h_stripe))])

        # # for i in range(int(finalL.shape[1] / h_stripe)):

        # distance[i,:,:] = np.array(np.sqrt(np.sum((desL[i][:, np.newaxis, :] - desR[i][np.newaxis, :, :]) ** 2, axis=-1)))

        # cv2.imshow('image_stripesL', totalimageL)

        # cv2.imshow('image_stripesR', totalimageR)

        # imgL = cv2.drawKeypoints(totalimageL, kptL[:], totalimageL, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # imgR = cv2.drawKeypoints(totalimageR, kptR[:], totalimageR, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # # cv2.imshow("skL", imgL)

        # # cv2.imshow("skR", imgR)

        # #     DETECT THE KPTS IN THE WHOLE IMAGES
        kpL, desL = sift.detectAndCompute(finalL, None)
        kpR, desR = sift.detectAndCompute(finalR, None)

        # draw key points detected

        # img2 = cv2.drawKeypoints(finalL, kp2, finalL, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.line(img2, (0, h_stripe), (img2.shape[1], h_stripe), (255,0,0))
        # cv2.imshow("grayframeL", img2)
        # img1 = cv2.drawKeypoints(finalR, kp1, finalR, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow("grayframeR", img1)
        n_stripes = int(finalL.shape[1] / h_stripe)
        kp_stripesL, des_stripesL, kp_stripesR, des_stripesR = keypoints_division(kpL, kpR, desL, desR, h_stripe, n_stripes)
        estimation_chessboard(frameL)

        outimg = np.concatenate((frameL, frameR), axis=1)

        disparity_map, n_kp = disparity_map_calculation(kp_stripesL, des_stripesL, kp_stripesR, des_stripesR)

            #     cv2.circle(finalL, (int(kp_stripesL[j, i[0]].pt[0]), int(kp_stripesL[j, i[0]].pt[1])), 4, (10 * i, 0, 0), 1)

            #     cv2.circle(finalR, (int(kp_stripesR[j, i[1]].pt[0]), int(kp_stripesR[j, i[1]].pt[1])), 4, (10 * i, 0, 0), 1)

            # temp_match = flann.knnMatch(des_stripesL[j], des_stripesR[j], k=2)
        # matches.append(temp_match)

        # ratio test as per Lowe's paper

        # for i, (m, n) in enumerate(temp_match):

        #     if m.distance < 0.8 * n.distance:

        #         good.append(m)


        if len(disparity_map) > 2:
            d_main = np.mean(disparity_map)
            # min_disp = d_main - 32
            # max_disp = d_main + 32
            disparity_map = np.array(disparity_map)
            disparity_map_star = np.ma.masked_inside(disparity_map, min_disp, max_disp)
            min_disp = d_main - (d_main / 2)
            max_disp = d_main + (d_main / 2)
            disparity_map_star = disparity_map[disparity_map_star.mask]
            std_dev = np.std(disparity_map_star)
            if std_dev > 0:  # (2 / dist_ob):
                val = filters.threshold_otsu(disparity_map_star)
                d_main_background = np.mean(disparity_map_star[disparity_map_star <= val])
                dist_bg = round(0.001 * focal_lenght * baseline / d_main_background, 2)
                d_main_object = np.mean(disparity_map_star[disparity_map_star > val])
                dist_ob = round(0.001 * focal_lenght * baseline / d_main_object, 2)
                if dist_bg - dist_ob < 0.1:
                    dist_bg = filtered_dist_bkg[0]

                filtered_dist_bkg[1] = dist_bg
                filtered_dist_obj[1] = dist_ob

                if abs(filtered_dist_obj[0] - filtered_dist_obj[1]) < treshold_o or first_o:
                    tot_dist.append(dist_ob)
                    filtered_dist_obj[0] = dist_ob
                    first_o = False
                else:
                    tot_dist.append(filtered_dist_obj[0])
                    count_o += 1
                if count_o == 10:
                    filtered_dist_obj[0] = dist_ob
                    count_o = 0

                if abs(filtered_dist_bkg[0] - filtered_dist_bkg[1]) < treshold_o or first_b:
                    filtered_dist_bkg[0] = dist_bg
                    tot_dist_bg.append(dist_bg)
                    first_b = False
                else:
                    tot_dist_bg.append(filtered_dist_bkg[0])
                    count_b += 1
                if count_b == 10:
                    filtered_dist_bkg[0] = dist_bg
                    count_b = 0
                count1 += 1
            else:
                d_main_object = np.mean(disparity_map_star)
                dist_ob = round(0.001 * focal_lenght * baseline / d_main_object, 2)
                tot_dist.append(filtered_dist_obj[0])
                tot_dist_bg.append(filtered_dist_bkg[0])
                count1 += 1
        else:
            count1 += 1
            tot_dist.append(dist_ob)
            tot_dist_bg.append(dist_bg)

        # cv2.putText(outimg, d_main, (10, 10), cv2.FONT_ITALIC, 2, 255)
        if n_frame < WS:
            tot_dist_filt.append(d0)
            tot_dist_bg_filt.append(d0)
        else:
            x = tot_dist[(n_frame - WS): n_frame]
            y = lfilter(FIR_coeff, 1, x)
            tot_dist_filt.append(y[-1])
            x_bg = tot_dist_bg[(n_frame - WS): n_frame]
            y_bg = lfilter(FIR_coeff, 1, x_bg)
            tot_dist_bg_filt.append(y_bg[-1])

        n_frame = n_frame + 1

        if y[-1] <= 0.8:
            warning = 'Warning: object too close!!!'
            cv2.putText(outimg, warning, (10, 50), cv2.FONT_ITALIC, 1, (0, 255, 255))
        # for i in range(int(min(des_stripesL.__len__(), des_stripesR.__len__()))):

        #     try:

        #         cv2.drawMatchesKnn(finalL, kp_stripesL[i], finalR, kp_stripesR[i], good[i], outimg)

        #         cv2.knn

        #     except(SystemError):

        #         continue

        # kpL = kp_stripesL[4]

        # kpR = kp_stripesR[4]

        # outimg = cv2.resize(outimg, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        info_dist = 'Distance from the obstacle: ' + str(round(y[-1], 2)) + 'm'
        info_dist_bg = 'Distance from the background: ' + str(round(y_bg[-1], 2)) + 'm'
        cv2.putText(outimg, info_dist, (10, 75), cv2.FONT_ITALIC, 1, (0, 255, 255))
        cv2.putText(outimg, info_dist_bg, (10, 100), cv2.FONT_ITALIC, 1, (0, 255, 255))

        cv2.imshow("outimg", outimg)
        # for i in range(int(min(des_stripesL.__len__(), des_stripesR.__len__()))):

        #     try:

        #         cv2.drawMatchesKnn(finalL, kp_stripesL[i], finalR, kp_stripesR[i], good[i], outimg)

        #         cv2.knn

        #     except(SystemError):

        #         continue

        # kpL = kp_stripesL[4]

        # kpR = kp_stripesR[4]

        # outimg = cv2.resize(outimg, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        cv2.imshow("outimg", outimg)
        # n_kp = np.sum(len(kp_struct[:]))

        # max_kp.append(n_kp)

        # d4 = d3

        # d3 = d2

        # d2 = d1

        # d1 = d0

        # d0 = temp_dim

        m = (30 - 60) / 27
        q = 60 - m * 3
        if 128 > d_main_object >= 0:
            if n_kp <= 3:
                temp_dim = 100
            elif 3 < n_kp < 30:
                temp_dim = int(m * n_kp + q)
            else:
                temp_dim = 50

        for i in range(len(dist_vect) - 1, 0, -1):
            dist_vect[i] = dist_vect[i - 1]
        dist_vect[0] = temp_dim
        dim_kp = int(np.mean(dist_vect))

        # print("n_kp: ", n_kp)

        # good = []

        # pts1 = []

        # pts2 = []

        #

        # # ratio test as per Lowe's paper

        # for i, (m, n) in enumerate(matches):

        #     if m.distance < 0.8 * n.distance:

        #         good.append(m)

        #         pts2.append(kp_stripesL[i][m.trainIdx].pt)

        #         pts1.append(kp_stripesR[i][m.queryIdx].pt)
    #     # # BFMatcher with default params

    #     # bf = cv2.BFMatcher(cv2.NORM_L2)

    #     # matches = bf.knnMatch(des1, des2, k=2)

    #     #

    #     # # Apply ratio test

    #     # good = []

    #     # for m, n in matches:

    #     #     if m.distance < 0.75 * n.distance:

    #     #         good.append([m])

    #

    #     # draw key points detected

    #     img1 = cv2.drawKeypoints(finalR, kp1, finalR, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #     cv2.imshow("grayframe1", img1)

    #     # distance = np.zeros((des1.shape[0], des2.shape[0]))

    #     # for i in range(des1.shape[0]-1):

    #     #     for j in range(des2.shape[0]-1):

    #     #         if kp1[i].pt[1] - kp2[j].pt[1] < 0.3 or kp1[i].pt[1] - kp2[j].pt[1] > -0.3:

    #     #             distance[i, j] = np.sqrt(np.sum((des1[i, :] - des2[j, :]) ** 2, axis = -1))

    #

    #     distance = np.array(np.sqrt(np.sum((des1[:, np.newaxis, :] - des2[np.newaxis, :, :]) ** 2, axis=-1)))

    #     #SIFT descriptor of a point is just 128-dimensional vector, so you can simple compute Euclidean distance

    #     between every two and match nearest pairs.

    #

    #     # for k in range(len(kp1)):

    #     #     for z in range(len(kp2)):

    #     #         if kp1[k].angle - kp2[z].angle > 0.5 or kp1[k].angle - kp2[z].angle < -0.5:

    #     #             distance[k, z] = 100000000000

    #     #         if kp1[k].pt[1] - kp2[z].pt[1] > 3 or kp1[k].pt[1] - kp2[z].pt[1] < -3:

    #     #             distance[k, z] = 100000000000

    #

    #     ind = np.unravel_index(np.argmin(distance, axis=None), distance.shape)

    #     print(distance.min())

    #     #ciccio = np.unravel_index(distance<100

    #     ind = tuple(zip(*np.where(distance < 50)))

    #     kpL = []

    #     kpR = []

    #     for i in ind:

    #         if 0.05 > kp2[i[1]].pt[1] - kp1[i[0]].pt[1] > -0.05 and 128 > kp2[i[1]].pt[0] - kp1[i[0]].pt[0] > 0:

    #             kpL.append(kp2[i[1]])

    #             kpR.append(kp1[i[0]])

    #     for i,j in enumerate(kpL):

    #         cv2.circle(centerL, (int(j.pt[0]), int(j.pt[1])), 4, (10*i, 3*i,25*i), 1)

    #     for i, j in enumerate(kpR):

    #         cv2.circle(centerR, (int(j.pt[0]), int(j.pt[1])), 4, (10*i, 3*i,25*i), 1)

    #     cv2.imshow('centerL', centerL)

    #

    #     cv2.imshow('centerR', centerR)

    #     # # create BFMatcher object

    #     #bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    #

    #     ## Match descriptors.

    #     #matches = bf.match(des1, des2)

    #

    #     ## Sort them in the order of their distance.

    #     #matches = sorted(matches, key=lambda x: x.distance)

    #     # good = []

    #     # for m, n in matches:

    #     #     if m.distance < 0.2 * n.distance:

    #     #         good.append([m])

    #     # # Draw first 10 matches.

    #     # img4 = np.hstack((centerL, centerR))

    #     # img3 = cv2.drawMatchesKnn(centerL, kp1, centerR, kp2, good, img4, flags=2)

    #

    #     #cv2.imshow('matches', img3)

    #

    #     cv2.imshow('FrameL', frameL)

    #     cv2.imshow('FrameR', frameR)

    # # Press Q on keyboard to  exit

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    # if k == 27:  # Esc key to stop

    #     break

    # elif k == -1:  # normally -1 returned,so don't print it

    #     continue

    # else:

    #     print(k)

    if cv2.waitKey(33) == ord('a'):
        outimg = cv2.resize(outimg, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("stopped_frame", outimg)
        while (True):
            k = cv2.waitKey(33)
            if k == 27:  # Esc key to stop
                break

capL.release()
capR.release()
cv2.destroyAllWindows()
# plt.plot(range(1, count1), tot_dist[1:])
# plt.plot(range(1, count1), tot_dist_bg[1:])
plt.plot(range(1, n_frame),tot_dist_bg_filt[1:])
plt.plot(range(1, n_frame),tot_dist_filt[1:])
plt.show()
