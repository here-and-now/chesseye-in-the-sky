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

        # #vertical lines

        for i in range(len(corners[0,:])):
            line_ = corners[:,i]
            x_ = line_[:,0]
            y_ = line_[:,1]
            a_, b_ = np.polyfit(x_, y_, 1)
            average_distance_ = np.mean(np.diff(x_,n=1))
            padded_line_ = np.pad(x_, (1,1), 'constant', constant_values=(x_[0]-average_distance_, x_[-1]+average_distance_))
            cv2.line(img, (int(padded_line_[0]), int(a_*padded_line_[0]+b_)), (int(padded_line_[-1]), int(a_*padded_line_[-1]+b_)), (0, 255, 0), 2)

        # horizontal lines
        for i in range(len(corners[:,0])):
            _line = corners[i,:]
            _x = _line[:,0]
            _y = _line[:,1]
            _a, _b = np.polyfit(_x, _y, 1)
            _average_distance = np.mean(np.diff(_x,n=1))
            _padded_line = np.pad(_x, (1,1), 'constant', constant_values=(_x[0]-_average_distance, _x[-1]+_average_distance))
            cv2.line(img, (int(_padded_line[0]), int(_a*_padded_line[0]+_b)), (int(_padded_line[-1]), int(_a*_padded_line[-1]+_b)), (0, 0, 255), 2)

    cv2.imshow('cv2_img', img)

    if cv2.waitKey(1) == 27:
        break







cv2.destroyAllWindows()




