import numpy as np
import cv2
from skimage import filters
import matplotlib.pyplot as plt
from scipy.signal import lfilter

sift = cv2.xfeatures2d.SIFT_create()

class Myfilter:
    def __init__(self, frame_rate, Cutoff_freq, d, order, FIR):
        # Define cutoff frequency
        self.frame_rate = frame_rate
        self.Cutoff_freq = Cutoff_freq
        self.d0 = d

        # Calculate Nyquist frequency
        self.Nyq_frequency = self.frame_rate / 2
        self.Cutoff_norm = self.Cutoff_freq / self.Nyq_frequency
        # FIR order
        self.order = order
        # Coefficients of the FIR filter
        self.FIR_coeff = FIR
        self.WS = len(self.FIR_coeff)
        self.x = list(2.6 * np.ones(self.WS))
        self.y = list(2.6 * np.ones(self.WS))
        self.x_bg = list(2.6 * np.ones(self.WS))
        self.y_bg = list(2.6 * np.ones(self.WS))

        self.tot_dist_filt = []
        self.tot_dist_bg_filt = []
        self.n_frame = 0

    def filtering(self):
        if self.n_frame < self.WS:
            self.tot_dist_filt.append(self.d0)
            self.tot_dist_bg_filt.append(self.d0)
        else:
            self.x = myrob.tot_dist[(self.n_frame - self.WS): self.n_frame]
            self.y = lfilter(self.FIR_coeff, 1, self.x)
            self.tot_dist_filt.append(self.y[-1])
            self.x_bg = myrob.tot_dist_bg[(self.n_frame - self.WS): self.n_frame]
            self.y_bg = lfilter(self.FIR_coeff, 1, self.x_bg)
            self.tot_dist_bg_filt.append(self.y_bg[-1])

        self.n_frame += 1


class Robot:
    def __init__(self, f_length, base, h_str):
        self.focal_lenght = f_length
        self.baseline = base
        self.h_stripe = h_str
        self.dim_kp = 30
        self.min_disp = 0
        self.max_disp = 64
        self.dist_ob = 2.6
        self.dist_bg = 2.8
        self.d_main_object = ((self.baseline * self.focal_lenght) / self.dist_ob) * 0.001
        self.d_main_background = ((self.baseline * self.focal_lenght) / self.dist_bg) * 0.001
        self.filtered_dist_obj = [self.dist_ob, self.dist_ob]
        self.filtered_dist_bkg = [self.dist_bg, self.dist_bg]
        self.treshold_o = 0.2
        self.tot_dist = []
        self.tot_dist_bg = []
        self.count_o = 0
        self.count_b = 0
        self.count1 = 0
        self.dist_vect = np.ones((5,), dtype='int') * self.dim_kp
        self.total_error_l = []
        self.total_error_h = []
        self.counter = 0
        self.total_l = []
        self.total_h = []

    def video_reading(self):
        capL = cv2.VideoCapture('robotL.avi')
        capR = cv2.VideoCapture('robotR.avi')

        if not capL.isOpened():
            print("Error opening video stream or file")

        if not capR.isOpened():
            print("Error opening video stream or file")
        return capL, capR

    def image_initialization(self, frameL, frameR):
        # Capture frame-by-frame
        # Display the resulting frame
        cv2.rectangle(frameL, (int(frameL.shape[1] / 2) - self.dim_kp, int(frameL.shape[0] / 2) - self.dim_kp),
                      (int(frameL.shape[1] / 2) + self.dim_kp, int(frameL.shape[0] / 2) + self.dim_kp), 250, 1)
        cv2.rectangle(frameR, (int(frameR.shape[1] / 2) - self.dim_kp, int(frameR.shape[0] / 2) - self.dim_kp),
                      (int(frameR.shape[1] / 2) + self.dim_kp, int(frameR.shape[0] / 2) + self.dim_kp), 250, 1)
        centerL = frameL[int(frameL.shape[0] / 2) - self.dim_kp: int(frameL.shape[0] / 2) + self.dim_kp,
                  int(frameL.shape[1] / 2) - self.dim_kp: int(frameL.shape[1] / 2) + self.dim_kp]
        centerR = frameR[int(frameR.shape[0] / 2) - self.dim_kp: int(frameR.shape[0] / 2) + self.dim_kp,
                  int(frameR.shape[1] / 2) - self.dim_kp:int(frameR.shape[1] / 2) + self.dim_kp]
        total_grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        total_intermediateL = cv2.equalizeHist(total_grayL)
        total_finalL = total_intermediateL
        grayL = cv2.cvtColor(centerL, cv2.COLOR_BGR2GRAY)
        intermediateL = cv2.equalizeHist(grayL)
        finalL = cv2.medianBlur(intermediateL, 5)

        grayR = cv2.cvtColor(centerR, cv2.COLOR_BGR2GRAY)
        intermediateR = cv2.equalizeHist(grayR)
        finalR = cv2.medianBlur(intermediateR, 5)
        return finalL, finalR

    def keypoints_division(self, kpL, kpR, desL, desR, n_stripes):

        kp_stripesL = []
        kp_stripesR = []
        des_stripesL = []
        des_stripesR = []

        for s in range(n_stripes):
            kp_stripesL.append([])
            kp_stripesR.append([])
            des_stripesL.append([])
            des_stripesR.append([])
            for j in range(len(kpL)):
                if s * self.h_stripe <= kpL[j].pt[1] < (s + 1) * self.h_stripe:
                    kp_stripesL[s].append(kpL[j])
                    des_stripesL[s].append(desL[j])
            for j in range(len(kpR)):
                if s * self.h_stripe <= kpR[j].pt[1] < (s + 1) * self.h_stripe:
                    kp_stripesR[s].append(kpR[j])
                    des_stripesR[s].append(desR[j])

        des_stripesL = np.array(des_stripesL)
        kp_stripesL = np.array(kp_stripesL)
        des_stripesR = np.array(des_stripesR)
        kp_stripesR = np.array(kp_stripesR)
        return kp_stripesL, des_stripesL, kp_stripesR, des_stripesR

    def estimation_chessboard(self, frameL, dist_ob, outimg):
        found, corners = cv2.findChessboardCorners(frameL, (6, 8))
        cv2.drawChessboardCorners(outimg, (6, 8), corners, found)

        if found:
            pt1 = corners[0]
            pt2 = corners[5]
            pt4 = corners[-1]
            pt = (corners[0], corners[-1])
            cv2.circle(frameL, (pt[0][0][0], pt[0][0][1]), 8, (0, 255, 0), 1)
            cv2.circle(frameL, (pt[1][0][0], pt[1][0][1]), 8, (0, 255, 0), 1)
            l_chess = pt2[0][0] - pt1[0][0]
            h_chess = pt4[0][1] - pt2[0][1]
            l_chess_mm = ((dist_ob * l_chess) / self.focal_lenght) * 1000
            h_chess_mm = ((dist_ob * h_chess) / self.focal_lenght) * 1000
            if l_chess_mm > 0 and h_chess_mm > 0:
                est_W = 'Estimate W: ' + str(round(l_chess_mm, 2)) + 'mm'
                est_H = 'Estimate H: ' + str(round(h_chess_mm, 2)) + 'mm'
                err = 'Error %: W ' + str(round(100*(l_chess_mm - 125) / 125, 2)) + '% | H ' + str(round(100*(h_chess_mm - 178) / 178, 2)) + '%'
                self.total_error_l.append(round(100*(l_chess_mm-125)/125, 2))
                self.total_error_h.append(round(100*(h_chess_mm-178)/178, 2))
                self.total_l.append(round(l_chess_mm, 2))
                self.total_h.append(round(h_chess_mm, 2))
                self.counter += 1
                cv2.putText(outimg, err, (650, 50), cv2.FONT_ITALIC, 1, (0, 255, 255))
                cv2.putText(outimg, est_W, (650, 75), cv2.FONT_ITALIC, 1, (0, 255, 255))
                cv2.putText(outimg, est_H, (650, 100), cv2.FONT_ITALIC, 1, (0, 255, 255))
        return outimg

    def disparity_map_calculation(self, kp_stripesL, des_stripesL, kp_stripesR, des_stripesR):
        distance = []
        disparity_map = []
        kp_struct = []

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
                    if removed[y] == 1:
                        for z in range(y + 1, len(ind)):
                            if ind[y][0] == ind[z][0] or ind[y][1] == ind[z][1]:
                                if distance[j][ind[y]] < distance[j][ind[z]]:
                                    if removed[z] == 1:
                                        index.remove(ind[z])
                                        removed[z] = 0
                                else:
                                    if removed[y] == 1:
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

    def distance_calculation(self, disparity_map):
        if len(disparity_map) > 2:
            d_main = np.mean(disparity_map)
            disparity_map = np.array(disparity_map)
            self.min_disp = d_main - (d_main / 2)
            self.max_disp = d_main + (d_main / 2)
            disparity_map_star = np.ma.masked_inside(disparity_map, self.min_disp, self.max_disp)
            disparity_map_star = disparity_map[disparity_map_star.mask]
            std_dev = np.std(disparity_map_star)

            if std_dev > 0.5 / self.dist_ob:
                val = filters.threshold_otsu(disparity_map_star)
                self.d_main_background = np.mean(disparity_map_star[disparity_map_star <= val])
                self.dist_bg = round(0.001 * self.focal_lenght * self.baseline / self.d_main_background, 2)
                self.d_main_object = np.mean(disparity_map_star[disparity_map_star > val])
                self.dist_ob = round(0.001 * self.focal_lenght * self.baseline / self.d_main_object, 2)
                self.dist_ob -= self.dist_ob * 0.065
                self.dist_bg -= self.dist_bg * 0.065
                if self.dist_bg - self.dist_ob < 0.1:
                    self.dist_bg = self.filtered_dist_bkg[0]

                self.filtered_dist_bkg[1] = self.dist_bg
                self.filtered_dist_obj[1] = self.dist_ob

                if (abs(self.filtered_dist_obj[0] - self.filtered_dist_obj[1]) < self.treshold_o):
                    self.tot_dist.append(self.dist_ob)
                    self.filtered_dist_obj[0] = self.dist_ob
                    self.count_o = 0
                else:
                    self.tot_dist.append(self.filtered_dist_obj[0])
                    self.count_o += 1

                if self.count_o == 10:
                    self.filtered_dist_obj[0] = self.dist_ob
                    self.count_o = 0

                if (abs(self.filtered_dist_bkg[0] - self.filtered_dist_bkg[1]) < self.treshold_o):
                    self.filtered_dist_bkg[0] = self.dist_bg
                    self.tot_dist_bg.append(self.dist_bg)

                    self.count_b = 0
                else:
                    self.tot_dist_bg.append(self.filtered_dist_bkg[0])
                    self.count_b += 1
                if self.count_b == 10:
                    self.filtered_dist_bkg[0] = self.dist_bg
                    self.count_b = 0
                self.count1 += 1
            else:
                self.tot_dist.append(self.filtered_dist_obj[0])
                self.tot_dist_bg.append(self.filtered_dist_bkg[0])
                self.count1 += 1
        else:
            self.count1 += 1
            self.tot_dist.append(self.dist_ob)
            self.tot_dist_bg.append(self.filtered_dist_bkg[0])

    def write_on_image(self, outimg):
        if myfilter.y[-1] <= 0.8:
            warning = 'Warning: object too close!!!'
            cv2.putText(outimg, warning, (10, 50), cv2.FONT_ITALIC, 1, (0, 255, 255))

        info_dist = 'Distance from the obstacle: ' + str(round(myfilter.y[-1], 2)) + 'm'
        info_dist_bg = 'Distance from the background: ' + str(round(myfilter.y_bg[-1], 2)) + 'm'
        cv2.putText(outimg, info_dist, (10, 75), cv2.FONT_ITALIC, 1, (0, 255, 255))
        cv2.putText(outimg, info_dist_bg, (10, 100), cv2.FONT_ITALIC, 1, (0, 255, 255))

        cv2.imshow("outimg", outimg)

        cv2.imshow("outimg", outimg)
        return outimg
    def square_dimention(self, n_kp):
        m = (20 - 60) / 57
        q = 60 - m * 3
        if 128 > self.d_main_object >= 0:
            if n_kp <= 3:
                temp_dim = 200
            elif 3 < n_kp < 60:
                temp_dim = int(m * n_kp + q)
            else:
                temp_dim = 20

        for i in range(len(self.dist_vect) - 1, 0, -1):
            self.dist_vect[i] = self.dist_vect[i - 1]
        self.dist_vect[0] = temp_dim
        dim_kp = int(np.mean(self.dist_vect))
        return dim_kp



myrob = Robot(567.2, 92.226, 2)
coeff = [-0.00545737100067199, 0.0317209689414059, 0.254972364809816, 0.437528074498901, 0.254972364809816, 0.0317209689414059, -0.00545737100067199]
myfilter = Myfilter(15, 0.5, 2.6, 6, coeff)
capL, capR = myrob.video_reading()
while capL.isOpened() and capR.isOpened():

    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if retL is True and retR is True:
        finalL, finalR = myrob.image_initialization(frameL, frameR)
        info_frame = np.empty_like(finalL)
        H = frameL.shape[0]
        L = frameL.shape[1]
        #
        h = finalL.shape[0]
        l = finalL.shape[1]
        #
        off_x_L = (L / 2 - l / 2)
        off_y_L = (H / 2 - h / 2)
        off_x_R = off_x_L + frameL.shape[1]
        off_y_R = off_y_L
        kpL, desL = sift.detectAndCompute(finalL, None)
        kpR, desR = sift.detectAndCompute(finalR, None)

        n_stripes = int(finalL.shape[1] / myrob.h_stripe)
        kp_stripesL, des_stripesL, kp_stripesR, des_stripesR = myrob.keypoints_division(kpL, kpR, desL, desR, n_stripes)
        outimg = np.concatenate((frameL, frameR), axis=1)
        outimg = myrob.estimation_chessboard(frameL, myrob.filtered_dist_obj[0], outimg)
        disparity_map, n_kp = myrob.disparity_map_calculation(kp_stripesL, des_stripesL, kp_stripesR, des_stripesR)
        myrob.distance_calculation(disparity_map)
        myfilter.filtering()
        outimg = myrob.write_on_image(outimg)
        myrob.dim_kp = myrob.square_dimention(n_kp)



    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    if cv2.waitKey(33) == ord('a'):
        outimg = cv2.resize(outimg, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("stopped_frame", outimg)
        while True:
            k = cv2.waitKey(33)
            if k == 27:  # Esc key to stop
                break

capL.release()
capR.release()
cv2.destroyAllWindows()

plt.figure()
plt.plot(range(1, myrob.count1), myrob.tot_dist_bg[1:],'g-')
plt.plot(range(1, myrob.count1), myrob.tot_dist[1:],'r-')
plt.gca().legend(('background distance','object distance'))
plt.title('Calculated distance')
plt.xlabel("Frame")
plt.ylabel("Distance")
plt.grid()
plt.show()
#
plt.figure()
plt.plot(range(1, myfilter.n_frame), myfilter.tot_dist_bg_filt[1:])
plt.plot(range(1, myfilter.n_frame), myfilter.tot_dist_filt[1:])
plt.gca().legend(('background distance','object distance'))
plt.title('Filtered distance')
plt.xlabel("Frame")
plt.ylabel("Distance")
plt.grid()
plt.show()

plt.figure()
plt.plot(range(1, myfilter.n_frame), myfilter.tot_dist_bg_filt[1:], range(1, myfilter.n_frame), myfilter.tot_dist_filt[1:], range(1, myrob.count1), myrob.tot_dist_bg[1:], range(1, myrob.count1), myrob.tot_dist[1:])
plt.gca().legend(('filtered background distance', 'filtered object distance', 'calculated background distance', 'calculated object distance'))
plt.title('Calculated vs filtered distance')
plt.xlabel("Frame")
plt.ylabel("Distance")
plt.grid()
plt.show()

plt.figure()
plt.plot(range(myrob.counter), myrob.total_error_l, '-y')
plt.plot(range(myrob.counter), myrob.total_error_h)
plt.plot(range(myrob.counter), (np.mean(myrob.total_error_l),)*myrob.counter, '-y')
plt.plot(range(myrob.counter), (np.mean(myrob.total_error_h),)*myrob.counter, '-b')

plt.gca().legend(('lenght error','height error'))
plt.title('Estimation error')
plt.xlabel("Frame")
plt.ylabel("% Error")
plt.grid()
plt.show()
print('mean height error', np.mean(myrob.total_error_h))
print('mean lenght error', np.mean(myrob.total_error_l))

t = range(myrob.counter)
data1 = myrob.total_l
data2 = myrob.total_h
data3 = myrob.total_error_l
data4 = myrob.total_error_h

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Frame')
ax1.set_ylabel('dimentions', color=color)
ax1.plot(t, data2, 'o-',color =color)
ax1.plot(t, data1, 'v-',color = color)

plt.gca().legend(('chessboard_height', 'chessboard_lenght'))
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('% Error', color=color)  # we already handled the x-label with ax1
ax2.plot(t, data4, 'o-',color = color)
ax2.plot(t, data3, 'v-', color= color)
plt.gca().legend(('height_error', 'lenght_error'))
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid()
plt.show()