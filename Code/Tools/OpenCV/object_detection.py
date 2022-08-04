'''
00:00 | Setup & Images
02:35 | Loading Template & Base Images
04:55 | Template Matching Methods
07:00 | Theory Behind Template Matching
14:10 | Displaying Matches
'''

import numpy as np
import cv2

# Read in the image to find and the image to find said image in
img = cv2.imread('figures/i_spy_0.jpg', 0)
template = cv2.imread('figures/i_spy_3.jpg', 0)
# Get the shape of the image to find
h, w = template.shape

# These are some prewritten functions that can be used to find the template
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED',
           'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 
           'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# Loop through the methods
for method in methods:
    img2 = img.copy()

    # (W - w + 1, H - h + 1)
    result = cv2.matchTemplate(img2, template, eval(method))
    # Min and max val of the result
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc

    # Draw a rectangle around the matched region
    bottom_right = (location[0] + w, location[1] + h)
    cv2.rectangle(img2, location, bottom_right, (255,0,0), 5)
    
    cv2.imshow("Match", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()