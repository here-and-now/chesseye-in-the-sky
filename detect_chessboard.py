import cv2
import numpy as np
import matplotlib.pyplot as plt



camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Detect inner chessboard corners
    plt.clf()

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

        # #horizontal lines
        for i in range(len(corners[:,0])):
            line = corners[i,:]
            #plt.plot(line[:,0], line[:,1])
            a, b = np.polyfit(line[:,0], line[:,1], 1)
            average_distance = np.mean(np.diff(line[:,0]))
            padded_line = np.pad(line[:,0], (1,1), 'constant', constant_values=(line[0,0]-average_distance, line[-1,0]+average_distance))
            plt.plot(padded_line, a*padded_line + b)
            # draw line on image
            cv2.line(img, (int(padded_line[0]), int(a*padded_line[0] + b)), (int(padded_line[-1]), int(a*padded_line[-1] + b)), (0,255,0), 2)


        # #vertical lines
        for i in range(len(corners[0,:])):
            line = corners[:,i]
            #plt.plot(line[:,0], line[:,1])
            a, b = np.polyfit(line[:,0], line[:,1], 1)
            # average_distance = np.mean(np.diff(line[:,0]))
            average_distance = np.mean(np.diff(line[:,0]))
            padded_line = np.pad(line[:,0], (1,1), 'constant', constant_values=(line[0,0]-average_distance, line[-1,0]+average_distance))
            plt.plot(padded_line, a*padded_line + b)
            # draw line on image
            cv2.line(img, (int(padded_line[0]), int(a*padded_line[0] + b)), (int(padded_line[-1]), int(a*padded_line[-1] + b)), (0,0,255), 2)





    plt.xlim(0, 640)
    plt.ylim(480, 0)
    #plt.show()

    cv2.imshow('cv2_img', img)

    if cv2.waitKey(1) == 27:
        break







cv2.destroyAllWindows()




