import cv2
import numpy as np
import matplotlib.pyplot as plt



camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, img = camera.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(gray,(5,5),0)
    ret,th = cv2.threshold(blur,200,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imshow('cv2_th', th)

    # Detect edges
    edges = cv2.Canny(th, 50, 150, apertureSize=3)

    cv2.imshow('cv2_edges', edges)

    # Detect lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 20)

    # Draw lines
    if lines is not None:
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imshow('cv2_board', img)




    # Draw detected corners
    if cv2.waitKey(1) == 27:
        break







cv2.destroyAllWindows()




