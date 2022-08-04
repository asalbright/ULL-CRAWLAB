# OpenCV Python Tutorial #4 - Drawing (Lines, Images, Circles & Text)
'''
00:00 | Introduction
01:01 | Drawing Lines
04:56 | Drawing Rectangles
08:00 | Drawing Text
13:35 | Outro
'''
import cv2
import numpy as np

cap = cv2.VideoCapture(0)   # number is the camera you want to access

while True:
    ret, frame = cap.read() # ret tells us if the capture worked, frame in a np.array of an image
    width = int(cap.get(3)) # 3 is one of the properties of cap.get() that is width
    height = int(cap.get(4)) # same as above, look up documentation for more on cap.get()
    
    # Shapes in an image
    start_pos = (0,0)
    end_pos = (width,height)
    color = (255, 0, 0)
    thickness = 10

    image = cv2.line(frame, start_pos, end_pos, color, thickness)
    image = cv2.line(frame, (0, height), (width, 0), (0,255,0), 10)
    image = cv2.rectangle(frame, (100, 100), (200, 200), (128, 128, 128), -1)
    image = cv2.circle(frame, (300, 300), 100, (0, 0, 255), 10)

    # text on an image
    position = (int(width/2), int(height/2))
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 1
    color = (0,150,150)
    thickness = 5

    image = cv2.putText(frame, 
                        "Check this out", 
                        position, 
                        font, 
                        size, 
                        color, 
                        thickness, 
                        cv2.LINE_AA) # LINE_AA is something that makes text look better

    cv2.imshow("Frame", frame) # show the frame

    if cv2.waitKey(1) == ord('q'): # if q is pressed quit
        break

cap.release()
cv2.destroyAllWindows()

