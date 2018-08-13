import cv2
import numpy as np
import pandas as pd


def Inhence_and_threshod(img,threshold):
    # lowpass filter
    gaussian = cv2.GaussianBlur(img, (3, 3), 1)

    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(gaussian)

    # THRESH
    __, th = cv2.threshold(cl, threshold, 255, cv2.THRESH_BINARY)

    return th


def inner_fill(th):
    # 接滿
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mor = cv2.dilate(th, kernel)

    # floodfill external's area
    image1, contours, hierarchy = cv2.findContours(
        mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mor, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # 侵蝕
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mor = cv2.erode(mor, kernel)
    mor = cv2.erode(mor, kernel)
    return mor


def draw(contours,img_color):


    # bone ROI
    for i in range(0, len(contours)):
        # 每一個輪廓往左移11

        for j in range(0,np.shape(contours[i])[0]):
            contours[i][j][0][0]=contours[i][j][0][0]+11


        area = cv2.contourArea(contours[i])
        x, y, w, h = cv2.boundingRect(contours[i])
        aspect_ratio = float(w) / h
        rect_area = w * h
        extent = float(area) / rect_area


        #delete error area
        if area>80:
           cv2.drawContours(img_color, [contours[i]], -1, (0, 0, 255), 1)


    #  bone number
    index = []
    centroid_x = []
    centroid_y = []
    i = 0
    for cnt in contours:
        # centroid
        M = cv2.moments(cnt)
        if M['m00']==0 :
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centroid_x = np.append(centroid_x, cx)
        centroid_y = np.append(centroid_y, cy)
        # index
        index = np.append(index, i)
        area = cv2.contourArea(cnt)
        if area>80:
         cv2.putText(img_color, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
         i += 1



def find_roi_coordinate(contours,img_color):
    df = pd.DataFrame()
    i = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        mask = np.zeros(img_color.shape, np.uint8)
        if area>80:
           cv2.drawContours(mask, [cnt], -1, 255, 1)
        pixelpoints = np.transpose(np.nonzero(mask))
        y = np.reshape(pixelpoints[:, 0], (len(pixelpoints[:, 0]), 1))
        x = np.reshape(pixelpoints[:, 1], (len(pixelpoints[:, 1]), 1))
        index = np.full(x.shape, i)
        pixelpoints = np.hstack((x, y, index))
        df_points = pd.DataFrame(pixelpoints, columns=['x', 'y', 'i'])
        df = df.append(df_points)
        i += 1
    #print(df)
    df.to_csv("coordinate.csv")
