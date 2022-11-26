import cv2
import chess
import time
import numpy as np

cam = cv2.VideoCapture(0)
while True:
    ret_val, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 2, 3, 0.08)
    dst = cv2.dilate(dst, None)

    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    print(dst)
    cv2.imshow('cv2_board', img)


    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()





