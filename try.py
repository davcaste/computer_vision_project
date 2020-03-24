import numpy as np
import cv2
from skimage import filters
from scipy.signal import kaiserord, lfilter, firwin, freqz

capL = cv2.VideoCapture('robotL.avi')
capR = cv2.VideoCapture('robotR.avi')
focal_lenght = 567.2  # in pixel
baseline = 92.226  # in mm
dim_kp = 30
h_stripe = 2
max_kp = []
# d4 = 2
# d3 = 2
# d2 = 2
# d1 = 2
# d0 = 2
dist = 3
dist_filter = [dist]
dist_vect = np.ones((5,), dtype='int')*dim_kp
d_main = ((baseline*focal_lenght)/dist)*0.001
d_main_object = d_main
d_main_foreground = d_main
dist_ob = dist
dist_fg = dist
temp_dim = dim_kp
flag = False
min_disp = 0
max_disp = 64

#------------------------------------------------
# Create a FIR filter and apply it to x.
#------------------------------------------------
sample_rate = 15

# The Nyquist rate of the signal.
nyq_rate = sample_rate / 2.0

vel = 0.45  # m/s
# The desired width of the transition from pass to stop,
# relative to the Nyquist rate.  We'll design the filter
# with a 5 Hz transition width.
width = 5.0/nyq_rate

# The desired attenuation in the stop band, in dB.
ripple_db = 60.0

# Compute the order and Kaiser parameter for the FIR filter.
N, beta = kaiserord(ripple_db, width)
# The cutoff frequency of the filter.
cutoff_hz = 0.10

# Use firwin with a Kaiser window to create a lowpass FIR filter.
taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

# Use lfilter to filter x with the FIR filter.
filtered_ob = lfilter(taps, 1.0, dist_vect)


if not capL.isOpened():
    print("Error opening video stream or file")

if not capR.isOpened():
    print("Error opening video stream or file")
sift = cv2.xfeatures2d.SIFT_create()

while capL.isOpened() and capR.isOpened():

    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if retL is True and retR is True:
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

        info_frame = np.empty_like(finalL)

        H = frameL.shape[0]
        L = frameL.shape[1]
        #
        h = finalL.shape[0]
        l = finalL.shape[1]

        #
        off_x_L = (L/2 - l/2)
        # int(L/2)
        off_y_L = (H/2 - h/2)
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
        # print('dentro')

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

        image_stripesL = np.zeros((int(finalL.shape[1] / h_stripe), h_stripe, finalL.shape[0]),
                                  dtype='uint8')  # NxHxL where N = n of stripes, H = height of stripes, L = lenght of window
        image_stripesR = np.zeros((int(finalR.shape[1] / h_stripe), h_stripe, finalR.shape[0]), dtype='uint8')
        kp_stripesL = []
        kp_stripesR = []
        des_stripesL = []
        des_stripesR = []
        # found, corners = cv2.findChessboardCorners(frameL, (6,8))
        # cv2.drawChessboardCorners(frameL, (6,8), corners, found)
        # if found == True:
        #     pt = (corners[0],corners[-1])
        #     cv2.circle(frameL,(pt[0][0][0],pt[0][0][1]),8,(0,255,0),1)
        #     cv2.circle(frameL,(pt[1][0][0],pt[1][0][1]),8,(0,255,0),1)
        #     l_chess = pt[1][0][0] - pt[0][0][0]
        #     h_chess = pt[1][0][1] - pt[0][0][1]
        #     l_chess_mm = (dist*l_chess)/focal_lenght
        #     h_chess_mm = (dist*h_chess)/focal_lenght
        #     print('ciao', l_chess_mm,h_chess_mm)
        #
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

        for i in range(int(finalL.shape[1] / h_stripe)):

            kp_stripesL.append([])
            kp_stripesR.append([])
            des_stripesL.append([])
            des_stripesR.append([])
            for j in range(len(kpL)):
                if i * h_stripe <= kpL[j].pt[1] < (i + 1) * h_stripe:
                    kp_stripesL[i].append(kpL[j])
                    des_stripesL[i].append(desL[j])
            for j in range(len(kpR)):
                if i * h_stripe <= kpR[j].pt[1] < (i + 1) * h_stripe:
                    kp_stripesR[i].append(kpR[j])
                    des_stripesR[i].append(desR[j])

        # print(len(des_stripesL))
        des_stripesL = np.array(des_stripesL)
        kp_stripesL = np.array(kp_stripesL)
        des_stripesR = np.array(des_stripesR)
        kp_stripesR = np.array(kp_stripesR)
        distance = []
        disparity_map = []
        kp_struct = []
        # distance = np.zeros((int(finalR.shape[1] / h_stripe),), dtype='object')
        # distance[i, :, :] = np.array(np.sqrt(np.sum((des_stripesL[i][:, np.newaxis, :] - des_stripesR[i][np.newaxis, :, :]) ** 2, axis=-1)))
        # distance = np.array(np.sqrt(np.sum((des1[:, np.newaxis, :] - des2[np.newaxis, :, :]) ** 2, axis=-1)))
        outimg = np.concatenate((frameL, frameR), axis=1)
        for j in range(len(des_stripesL)):
            des_stripesL[j] = np.array(des_stripesL[j])
            kp_stripesL[j] = np.array(kp_stripesL[j])
            des_stripesR[j] = np.array(des_stripesR[j])
            kp_stripesR[j] = np.array(kp_stripesR[j])
            distance.append([])
            # print(kp_stripesL[j].__len__(), " ", kp_stripesR[j].__len__())
            if kp_stripesL[j].__len__() > 0 and kp_stripesR[j].__len__() > 0:
                distance[j] = np.array(np.sqrt(np.sum((des_stripesL[j][:, np.newaxis, :] - des_stripesR[j][np.newaxis, :, :]) ** 2, axis=-1)))

                ind = tuple(zip(*np.where(distance[j] < 150)))       ### GUARDA QUI!!!!!!!!!!!!!!!
                index = list(ind)
                # print(ind)
                # print(index)
                removed = np.ones(len(ind), dtype='uint8')

                for i in range(len(ind)-1):
                    check = False
                    if removed[i] == 1:
                        for k in range(i+1, len(ind)):
                            if ind[i][0] == ind[k][0] or ind[i][1] == ind[k][1]:
                                if distance[j][ind[i]] < distance[j][ind[k]]:
                                    if removed[k] == 1:
                                        # print(ind[k])
                                        index.remove(ind[k])
                                        removed[k] = 0
                                else:
                                    if removed[i] == 1:
                                        # print('ciao', ind[i])
                                        index.remove(ind[i])
                                        removed[i] = 0

                index = tuple(index)
                # print(removed)
                # print('ind after cleaning',ind)
                # print('ind after cleaning',index)
                for i in index:
                    uL = int(kp_stripesL[j][i[0]].pt[0] + off_x_L)
                    uR = int(kp_stripesR[j][i[1]].pt[0] + off_x_R)
                    vL = int(kp_stripesL[j][i[0]].pt[1] + off_y_L)
                    vR = int(kp_stripesR[j][i[1]].pt[1] + off_y_R)
                    disparity = uL - uR
                    # print("(", uL, ", ",  vL, ") ", "(", uR, ", ",  vR, ") ", disparity)
                    kp_struct.append(((uL, vL), (uR, vR)))
                    if disparity + frameL.shape[1] > 0:
                        disparity_map.append(disparity + frameL.shape[1])
                        cv2.circle(outimg, (uL, vL), 3, (255, 0, 0), 1)
                        cv2.circle(outimg, (uR, vR), 3, (255, 0, 0), 1)
                        cv2.line(outimg, (uL, vL), (uR, vR), (255, 0, 0), 1)

            #     cv2.circle(finalL, (int(kp_stripesL[j, i[0]].pt[0]), int(kp_stripesL[j, i[0]].pt[1])), 4, (10 * i, 0, 0), 1)
            #     cv2.circle(finalR, (int(kp_stripesR[j, i[1]].pt[0]), int(kp_stripesR[j, i[1]].pt[1])), 4, (10 * i, 0, 0), 1)
            #temp_match = flann.knnMatch(des_stripesL[j], des_stripesR[j], k=2)
           # matches.append(temp_match)
            # ratio test as per Lowe's paper
            # for i, (m, n) in enumerate(temp_match):
            #     if m.distance < 0.8 * n.distance:
            #         good.append(m)
        n_kp = len(kp_struct)
        # print(n_kp)

        fig = None
        if len(disparity_map) > 2:
            d_main = np.mean(disparity_map)
            # print(d_main)
            # min_disp = d_main - 32
            # max_disp = d_main + 32

            disparity_map = np.array(disparity_map)
            #disparity_map_star = disparity_map < max_disp
            #disparity_map_star = disparity_map > min_disp

            disparity_map_star = np.ma.masked_inside(disparity_map,min_disp,max_disp)
            min_disp = d_main - (d_main / 2)
            max_disp = d_main + (d_main / 2)
            #disp_tot = (disparity_map_star.all() == disparity_map_star1.all())
            disparity_map_star = disparity_map[disparity_map_star.mask]
            std_dev = np.std(disparity_map_star)
            print(std_dev)
            print(2/dist_ob)
            if std_dev > (2/dist_ob):
                val = filters.threshold_otsu(disparity_map_star)
                # print(val)
                d_main_background = np.mean(disparity_map_star[disparity_map_star <= val])
                dist_bg = round(0.001 * focal_lenght * baseline / d_main_background, 2)
                #filtered_bg = lfilter(taps, 1.0, dist_bg)
                print("d_main_background: ", round(d_main_background, 2), 'distance_background: ', dist_bg, "m")
                d_main_object = np.mean(disparity_map_star[disparity_map_star > val])
                dist_ob = round(0.001 * focal_lenght * baseline / d_main_object, 2)
                dist_filter.append(dist_ob)
                if len(dist_filter)>12:
                    print('fatto')
                    dist_filter=dist_filter[len(dist_filter)-12:-1]
                    print(dist_filter)
                filtered_ob = lfilter(taps, 1.0, dist_filter)
                print("d_main_object: ", round(d_main_object, 2), 'distance_object: ', filtered_ob[-1], "m")
            else:
                d_main_object = np.mean(disparity_map_star)
                dist_ob = round(0.001 * focal_lenght * baseline / d_main_object, 2)
                dist_filter.append(dist_ob)
                filtered_ob = lfilter(taps, 1.0, dist_filter)
                print("d_main: ", round(d_main_object, 2), 'distance: ', filtered_ob[-1], "m")
                flag = False
        else:

            if flag == False:
                print("d_main ~ ", round(d_main_object, 2), 'distance ~ ', filtered_ob[-1], "m")
            else:
                print("d_main <= ", round(d_main_object, 2), 'distance <= ', filtered_ob[-1], "m")
            flag = True
        # cv2.putText(outimg, d_main, (10, 10), cv2.FONT_ITALIC, 2, 255)

        # for m in range(len(kp_struct)):
        #         cv2.line(outimg, kp_struct[m][0], kp_struct[m][1], (255, 0, 0), 1)
        if filtered_ob[-1] <= 0.8:
            print('Warning: object too close')

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
        #n_kp = np.sum(len(kp_struct[:]))
        #max_kp.append(n_kp)

        # d4 = d3
        # d3 = d2
        # d2 = d1
        # d1 = d0
        # d0 = temp_dim


        m = (30 - 60)/25
        q = 60 - m*5
        if 128 > d_main_object >= 0:
            if n_kp <= 5:
                temp_dim = 100
            elif 5 < n_kp <30:
                temp_dim = int(m*n_kp+q)
            else:
                temp_dim = 50
        print('initial dist vect',dist_vect)
        # dist_vect[4] = dist_vect[3]
        # dist_vect[3] = dist_vect[2]
        # dist_vect[2] = dist_vect[1]
        # dist_vect[1] = dist_vect[0]
        # dist_vect[0] = temp_dim
        for i in range(len(dist_vect)-1, 0, -1):
            dist_vect[i] = dist_vect[i-1]
        dist_vect[0] = temp_dim
        print(temp_dim)
        print('final dist vect',dist_vect)
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
        outimg = cv2.resize(outimg, None, fx = 1.5, fy = 1.5, interpolation = cv2.INTER_CUBIC)
        cv2.imshow("stopped_frame", outimg)
        while(True):
            k = cv2.waitKey(33)
            if k == 27:  # Esc key to stop
                break

capL.release()
capR.release()

cv2.destroyAllWindows()

