import cv2
import chess
import time
import numpy as np
import scipy.cluster as clstr
import scipy.spatial as spatial
from collections import defaultdict
from functools import partial


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

def intersections(h, v):
    points = []
    h = np.reshape(h, (-1, 2))
    v = np.reshape(v, (-1, 2))

    for rho_h, theta_h in h:
        for rho_v, theta_v in v:
            A = np.array([[np.cos(theta_h), np.sin(theta_h)], [np.cos(theta_v), np.sin(theta_v)]])
            b = np.array([rho_h, rho_v])
            point = np.linalg.solve(A, b)
            points.append(point)
    return np.array(points)


def cluster(points, max_dist=5):
    Y = spatial.distance.pdist(points)
    Z = clstr.hierarchy.single(Y)
    T = clstr.hierarchy.fcluster(Z, max_dist, 'distance')
    clusters = defaultdict(list)
    for i in range(len(T)):
        clusters[T[i]].append(points[i])
    clusters = clusters.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:,0]), np.mean(np.array(arr)[:,1])), clusters)
    return clusters


def closest_point(points, loc):
    dists = np.array(map(partial(spatial.distance.euclidean, loc), points))
    closest_point = points[np.argmin(dists)]
    return closest_point




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

        # Find intersections
        points = intersections(h_lines, v_lines)
        # Draw intersections
        # for point in points:
        #     cv2.circle(img, tuple(point.astype(int)), 5, [0, 255, 0], -1)


        # from list of points find 81 most inner points
        # https://stackoverflow.com/questions/48954246/find-sudoku-grid-using-opencv-and-python
        # find the center of the image
        center = np.array([img.shape[1] / 2, img.shape[0] / 2])
        # draw center point
        cv2.circle(img, tuple(center.astype(int)), 5, [0, 255, 0], -1)
        # find the closest point to the center
        center_point = closest_point(points, center)
        # find the closest points to the center point
        points = np.array(points)
        dists = np.array(map(partial(spatial.distance.euclidean, center_point), points))
        closest_points = points[dists.argsort()[:81]]
        # draw the closest points
        for point in closest_points:
            cv2.circle(img, tuple(point.astype(int)), 5, [0, 0, 255], -1)

        cv2.imshow('cv2_board', img)


    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()





