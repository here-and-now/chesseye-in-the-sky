import cv2
import numpy as np
import chess
import time

cam = cv2.VideoCapture(0)

while True:
    ret_val, image = cam.read()

    # find chessboard squares and draw them on the image
    # do not use findchessboardcorners functions because it wont work

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blur the image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold the image
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

    # find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # find the biggest contour (if detected)
    max_area = 0
    ci = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            ci = i

    # draw the biggest contour
    cnt = contours[ci]
    hull = cv2.convexHull(cnt)
    cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)
    cv2.drawContours(image, [hull], 0, (0, 0, 255), 2)

    # find the convex hull
    hull = cv2.convexHull(cnt, returnPoints=False)

    # find the convexity defects
    defects = cv2.convexityDefects(cnt, hull)

    # find the number of defects

    # find the number of defects
    if defects is not None:
        num_defects = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
            if angle <= np.pi / 2:
                num_defects += 1
                cv2.circle(image, far, 1, [0, 0, 255], -1)
            cv2.line(image, start, end, [0, 255, 0], 2)

        if num_defects == 0:
            cv2.putText(image, "ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif num_defects == 1:
            cv2.putText(image, "TWO", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        elif num_defects == 2:
            cv2.putText(image, "THREE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif num_defects == 3:
            cv2.putText(image, "FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

    cv2.imshow('image', image)

    if cv2.waitKey(1) == 27:
        break




cv2.destroyAllWindows()

