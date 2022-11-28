import cv2
import numpy as np
import matplotlib.pyplot as plt



camera = cv2.VideoCapture(0)

while True:
    # Detect inner chessboard corners
    ret, img = camera.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

    # Draw detected corners
    if ret:
        corners = np.squeeze(corners)
        cv2.drawChessboardCorners(img, (7, 7), corners, ret)

        # transform 49,2 array to 7,7,2 array
        corners = np.reshape(corners, (7,7,2))
        # expand 7,7,2 array to 8,8,2 array
        #corners = np.pad(corners, ((1,1),(1,1),(0,0)), 'constant', constant_values=0)

        line1 = corners[1]
        plt.plot(line1[:,0], line1[:,1])

        a, b = np.polyfit(line1[:,0], line1[:,1], 1)

        average_distance = np.mean(np.diff(line1[:,0]))
        padded_line = np.pad(line1[:,0], (1,1), 'constant', constant_values=(line1[0,0]-average_distance, line1[-1,0]+average_distance))


        plt.plot(line1[:,0], a*line1[:,0] + b)

        # plot fitted line,




    plt.xlim(0, 640)
    plt.ylim(480, 0)
    plt.show()

    #cv2.imshow('cv2_img', img)

    if cv2.waitKey(1) == 27:
        break







cv2.destroyAllWindows()




