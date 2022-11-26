import cv2
import chess
import time
import numpy as np



# find chessboard corners from webcam feed
def find_chessboard_corners():
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        ret, corners = cv2.findChessboardCorners(img, (7, 7))

        # if ret:
        cv2.drawChessboardCorners(img, (7, 7), corners, ret)
        cv2.imshow('Board', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()



# corner detection with harris corner detection
def harris_corner_detection():
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.08)
        dst = cv2.dilate(dst, None)
        img[dst > 0.01 * dst.max()] = [0, 0, 255]
        cv2.imshow('Board', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()
harris_corner_detection()


# draw chessboard field names as a1, b1, c1, etc. on webcam feed
def draw_field_names():
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()

        for i in range(8):
            for j in range(8):
                cv2.putText(img, chess.square_name(i * 8 + j), (i * 50 + 50, j * 50 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('Board', img)
        time.sleep(1)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()






# draw_field_names()
# find_chessboard_corners()




