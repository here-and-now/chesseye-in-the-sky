# modified from https://stackoverflow.com/questions/48954246/find-sudoku-grid-using-opencv-and-python

import cv2
import chess
import time

import matplotlib.pyplot as plt
import numpy as np

cam = cv2.VideoCapture(0)

# Set camera resolution
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
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
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img

filter = True

while True:
    ret_val, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 0, 250, apertureSize=3)
    cv2.imshow('cv22edges', edges)

    # kernel = np.ones((3, 3), np.uint8)
    # edges = cv2.dilate(edges, kernel, iterations=1)
    # cv2.imshow('cv22edges2', edges)

    # kernel = np.ones((3, 3), np.uint8)
    # edges = cv2.erode(edges, kernel, iterations=1)
    # cv2.imshow('cv22edges3', edges)


    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is None:
        print('No lines found')
        continue

    # Draw lines
    img = draw_lines(img, lines, color=[0, 0, 255], thickness=1)

    plt.clf()
    if filter:



        rho_threshold = 30
        theta_threshold = 0.5

        # how many lines are similar to a given one
        similar_lines = {i : [] for i in range(len(lines))}


        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue

                rho_i,theta_i = lines[i][0]
                rho_j,theta_j = lines[j][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    similar_lines[i].append(j)

        # ordering the INDECES of the lines by how many are similar to them
        indices = [i for i in range(len(lines))]
        indices.sort(key=lambda x : len(similar_lines[x]))

        

        # line flags is the base for the filtering
        line_flags = len(lines)*[True]
        for i in range(len(lines) - 1):
            # if we already disregarded the ith element in the ordered list then we
            # don't care (we will not delete anything based on it and we will never
            # reconsider using this line again)
            if not line_flags[indices[i]]: 
                continue
            # we are only considering those elements that had less similar line
            for j in range(i + 1, len(lines)): 
                # and only if we have not disregarded them already
                if not line_flags[indices[j]]: 
                    continue

                rho_i,theta_i = lines[indices[i]][0]
                rho_j,theta_j = lines[indices[j]][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    # if it is similar and have not been disregarded yet then drop it now
                    line_flags[indices[j]] = False

        print('number of Hough lines:', len(lines))

        filtered_lines = []
        if filter:
            for i in range(len(lines)):
                if line_flags[i]:
                    filtered_lines.append(lines[i])
            print('Number of filtered lines:', len(filtered_lines))
        else:
            filtered_lines = lines

         # list of angles

        # theta_list = [[theta for rho, theta in line] for line in filtered_lines]
        # mean = np.mean(theta_list)
        # # split into two groups
        # theta_list1 = [theta for theta in theta_list if theta < mean]
        # theta_list2 = [theta for theta in theta_list if theta > mean]
        #
        # # find median of each group
        # median1 = np.median(theta_list1)
        # median2 = np.median(theta_list2)
        #
        #
        # # exclude outliers in each group
        # theta_list1 = [theta for theta in theta_list1 if abs(theta - median1) < 0.1]
        # theta_list2 = [theta for theta in theta_list2 if abs(theta - median2) < 0.1]
        #
        # # conctruct new filtered line list based on the two groups
        # filtered_lines = []
        # for line in lines:
        #     rho, theta = line[0]
        #     if theta in theta_list1 or theta in theta_list2:
        #         filtered_lines.append(line)


        # Draw lines
        # img = draw_lines(img, filtered_lines, color=[0, 255, 0], thickness=1)
        img = draw_lines(img, filtered_lines, color=[255, 0, 0], thickness=2)



        cv2.imshow('cv2_board', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()





