# OpenCV Python Tutorial #3 - Cameras and VideoCapture
'''
00:00 | Introduction
00:50 | Displaying video capture device
07:11 | Mirroring video multiple times
'''
import cv2
import numpy as np

cap = cv2.VideoCapture(1)   # number is the camera you want to access

while True:
    ret, frame = cap.read() # ret tells us if the capture worked, frame in a np.array of an image
    width = int(cap.get(3)) # 3 is one of the properties of cap.get() that is width
    height = int(cap.get(4)) # same as above, look up documentation for more on cap.get()

    image = np.zeros(frame.shape, np.uint8) # create an array to fill
    sm_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) # shrink the image
    image[:height//2, :width//2] = cv2.rotate(sm_frame, cv2.ROTATE_180)
    image[height//2:, width//2:] = sm_frame
    image[:height//2, width//2:] = sm_frame
    image[height//2:, :width//2] = sm_frame
    

    cv2.imshow("Frame", image) # show the frame

    if cv2.waitKey(1) == ord('q'): # if q is pressed quit
        break

cap.release()
cv2.destroyAllWindows()
