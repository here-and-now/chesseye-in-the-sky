import cv2
import chess
import time
import numpy as np

cam = cv2.VideoCapture(0)

# Set camera resolution
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret_val, img = cam.read()

    # Hough line detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 0, 250, apertureSize=3)
    cv2.imshow('cv22edges', edges)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
    line_list = []
    if lines is not None:
        # get most common angle
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
            # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # convert to degrees
            deg = theta * 180 / np.pi
            line_list.append([rho, deg])

            #HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) -> lines


    # export lines to csv
    np.savetxt("lines.csv", line_list, delimiter=",", fmt="%10.4f")

    # find most common angle in list with allowed deviation of 1 degrees

    most_common_angle = 0
    most_common_angle_count = 0
    for line in line_list:
        angle = line[1]
        angle_count = 0
        for line2 in line_list:
            if abs(line2[1] - angle) < 2:
                angle_count += 1
        if angle_count > most_common_angle_count:
            most_common_angle_count = angle_count
            most_common_angle = angle

    print(most_common_angle, most_common_angle_count)

    # draw lines with most common angle and deviation of 1 degrees

    for line in lines:
        rho, theta = line[0]
        if abs(theta - np.radians(most_common_angle)) < np.radians(1):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)


    cv2.imshow('cv2_board', img)


    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()





