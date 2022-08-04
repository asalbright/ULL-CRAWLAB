# OpenCV Python Tutorial #6 - Corner Detection
'''
Relevant Documentation: https://opencv-python-tutroals.readth...

00:00 | Introduction
01:37 | Corner detection
09:42 | Drawing corners
15:04 | Drawing lines between corners
18:35 | Outro 
'''

import cv2
import numpy as np

img = cv2.imread("figures/fancySquares.jpg")
img = cv2.resize(img, (0,0), fx=1.5, fy=1.5)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

num_corners = 25
quality = 0.005 # between 0 and 1, its the confidence of the quality
min_dist = 1 # absolute distance between two corners, this way you dont get a bunch of corers on one round corner

corners = cv2.goodFeaturesToTrack(img_gray, num_corners, quality, min_dist) # corners are float point values
corners = np.int0(corners) # this will take the np.array that is corners and make all the vals ints

# Draw a circle on every corner detected
for corner in corners:
    x, y = corner.ravel() # this flattens and array [[x, y]] -> [x, y]
    cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
    print(corner)

# Draw lines between all the different corners on the screen with random colors
for i in range(len(corners)):
    for j in range(i+1, len(corners)):
        corner1 = tuple(corners[i][0])
        corner2 = tuple(corners[j][0])
        color = tuple(map(lambda x: int(x), np.random.randint(0, 255, size=3)))
        cv2.line(img, corner1, corner2, color, 1)

        
cv2.imshow('Frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()