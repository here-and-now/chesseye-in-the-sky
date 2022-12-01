import cv2
import chess
import time
import numpy as np
import scipy as sc

cam = cv2.VideoCapture(0)

# Set camera resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    ret_val, img = cam.read()

    # create empty img copy
    empty = np.zeros(img.shape, np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 2, 3, 0.08)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    # here u can get corners
    print(corners)

    # Now draw them


    res = np.hstack((centroids, corners))
    res = np.int0(res)
    img[res[:, 1], res[:, 0]] = [0, 0, 255]
    img[res[:, 3], res[:, 2]] = [0, 255, 0]

    # empty[res[:, 1], res[:, 0]] = [0, 0, 255]
    empty[res[:, 3], res[:, 2]] = [0, 255, 0]

    # hough line detection

    gray = cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    edges = cv2.Canny(gray, 0, 255, apertureSize=3)
    cv2.imshow('edges', edges)
    lines = cv2.HoughLines(edges, 2, np.pi / 180, 30)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)




    cv2.imshow('cv2_empty', empty)




    cv2.imshow('cv2_board', img)


    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()





