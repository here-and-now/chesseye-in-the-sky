import cv2
import numpy as np
import chess
import time

cam = cv2.VideoCapture(0)

while True:
    ret_val, image = cam.read()
    #resize image
    image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
    mask = np.zeros(image.shape, dtype=np.uint8)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    cv2.imshow('cv2_blur', blur)
    # blur = gray


    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imshow('cv2_thresh', thresh)

    # Remove noise with morph operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # invert image
    invert = 255 - opening
    cv2.imshow('cv2_invert', invert)

    # Find contours and find squares with contour area filtering + shape approximation
    # contours, hierarchy = cv2.findContours(invert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(invert, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    contours, hierarchy = cv2.findContours(invert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if len(approx) == 4 and area > 30 and area < 10000:
        if True:
            x,y,w,h = cv2.boundingRect(c)
            cv2.drawContours(original, [c], -1, (36,255,12), 2)
            cv2.drawContours(mask, [c], -1, (255,255,255), -1)
            # cv2.drawContours(mask, [c], -1, (0,0,0), -1)



    cv2.imshow("cv2_original", original)
    cv2.imshow("cv2_mask", mask)

    # time.sleep(0.5)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
