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


def auto_canny(image, sigma=0.33):
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged
def split_into_h_v_lines(lines):
    h_lines = []
    v_lines = []
    for line in lines:
        rho, theta = line[0]
        if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
            v_lines.append(line)
        else:
            h_lines.append(line)
    return h_lines, v_lines

def filter_lines(lines, rho_threshold=0.1, theta_threshold=np.pi / 180):
    #https: // stackoverflow.com / questions / 48954246 / find - sudoku - grid - using - opencv - and -python
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
        if not line_flags[indices[i]]:
            continue
        for j in range(i + 1, len(lines)):
            if not line_flags[indices[j]]:
                continue
            rho_i,theta_i = lines[indices[i]][0]
            rho_j,theta_j = lines[indices[j]][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                line_flags[indices[j]] = False
    filtered_lines = []
    if filter:
        for i in range(len(lines)):
            if line_flags[i]:
                filtered_lines.append(lines[i])
    else:
        filtered_lines = lines

    return filtered_lines


while True:
    ret_val, img = cam.read()

    # Hough line detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = auto_canny(gray)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    cv2.imshow('cv2_dilate', edges)

    # erode to remove noise
    kernel = np.ones((4, 4), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)
    cv2.imshow('cv2_erode', edges)

    # Hough lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180 / 4, 250)
    if lines is not None:

        # filter lines that are close together
        lines = filter_lines(lines, rho_threshold=20, theta_threshold=0.1)
        # img = draw_lines(img, lines, [255, 255, 255], thickness=1)

        # Split lines into horizontal and vertical
        h_lines, v_lines = split_into_h_v_lines(lines)
        # Draw lines
        img = draw_lines(img, v_lines, [0, 0, 255], thickness=1)
        img = draw_lines(img, h_lines, [0, 255, ], thickness=1)


        cv2.imshow('cv2_board', img)


    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()





