import cv2
import chess
import time
import numpy as np

cam = cv2.VideoCapture(0)

# Set camera resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    ret_val, img = cam.read()

    # Hough line detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 150, 250, apertureSize=3)

    #HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) -> lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, minLineLength=200, maxLineGap=100)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)



    # cluster lines that are which are withing 20 pixels of each other
    # and are within 10 degrees of each other


    cv2.imshow('cv2_board', img)


    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()





