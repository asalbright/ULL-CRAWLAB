'''
Script for saving screenshots of images.

Use aswd keys to move around.
Use +/- withought the shift key to resize box.
use c key to capture image.
use q to quit application and close windows.

Note:
Do not move box out of frame or resize box out of frame.
This may crash program.
'''

import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
import sys


def main():
    x1 = 0
    x2 = 100
    y1 = 0
    y2 = 100
    while(True):

        img = cv2.imread("figures/fancySquares.jpg")
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 256), 1)
        cv2.imshow("image", img)

        key = cv2.waitKey(1)
        if key == ord('d'):
            x1 += 5
            x2 += 5
        elif key == ord('a'):
            x1 -= 5
            x2 -= 5
        if key == ord('w'):
            y1 -= 5
            y2 -= 5
        elif key == ord('s'):
            y1 += 5
            y2 += 5
        if key == ord('='):
            x1 -= 5
            y1 -= 5
            x2 += 5
            y2 += 5
        elif key == ord('-'):
            x1 += 5
            y1 += 5
            x2 -= 5
            y2 -= 5

        if key == ord('q'): break
        elif key == ord('c'):
            save_image(img, x1, x2, y1, y2)

        print(f"x1, y1 = {(x1, y1)}")
        print(f"x2, y2 = {(x2, y2)}")

    cv2.destroyAllWindows()

def save_image(img, x1, x2, y1, y2):
    save = img[y1+1:y2, x1+1:x2]
    cv2.imwrite(f"image_{x1}{y1}{x2}{y2}.jpg", save)

if __name__ == "__main__":
    main()