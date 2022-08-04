# OpenCV Python Tutorial #5 - Colors and Color Detection
'''
00:00 | Introduction
00:45 | HSV Color
06:40 | Masks
'''
import cv2
import numpy as np

cap = cv2.VideoCapture(0)   # number is the camera you want to access

while True:
    ret, frame = cap.read() # ret tells us if the capture worked, frame in a np.array of an image
    width = int(cap.get(3)) # 3 is one of the properties of cap.get() that is width
    height = int(cap.get(4)) # same as above, look up documentation for more on cap.get()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # convert the image to a HSV image
    upper_yellow = np.array([30, 255, 255]) # BGR not RGB
    lower_yellow = np.array([20, 100, 100]) 

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow) # typically is something that can be used to select parts of an image to keep
    result = cv2.bitwise_and(frame, frame, mask=mask) # there are other ways to do this

    cv2.imshow("Frame", result) # show the frame
    cv2.imshow("Mask", mask) # show the mask, notice it is all binary (black/white)

    if cv2.waitKey(1) == ord('q'): # if q is pressed quit
        break

cap.release()
cv2.destroyAllWindows()

BGR_color = np.array([[[255, 0, 0]]])
# to get one pixel just print the next statemtnt and pick a single value
cv2.cvtColor(BGR_color, cv2.COLOR_BGR2HSV) # cvtColor expects an array



# Found this on Stack Overflow:
def colorDetection(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    '''Red'''
    # Range for lower red
    red_lower = np.array([0,120,70])
    red_upper = np.array([10,255,255])
    mask_red1 = cv2.inRange(hsv, red_lower, red_upper)

    # Range for upper range
    red_lower = np.array([170,120,70])
    red_upper = np.array([180,255,255])
    mask_red2 = cv2.inRange(hsv, red_lower, red_upper)

    mask_red = mask_red1 + mask_red2

    red_output = cv2.bitwise_and(image, image, mask=mask_red)

    red_ratio=(cv2.countNonZero(mask_red))/(image.size/3)

    print("Red in image", np.round(red_ratio*100, 2))



    '''yellow'''
    # Range for upper range
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

    yellow_output = cv2.bitwise_and(image, image, mask=mask_yellow)

    yellow_ratio =(cv2.countNonZero(mask_yellow))/(image.size/3)

    print("Yellow in image", np.round(yellow_ratio*100, 2))