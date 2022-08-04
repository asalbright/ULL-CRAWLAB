# ********************************************
# Author: Andrew Albright
# Date: 03/31/2021

# File containing useful functions

# ********************************************

import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import pandas
import os
import sys
import cv2
from threading import Thread

class VideoWrite():
    def __init__(self, frame_width=640, frame_height=480, fps=60, file_name="Video_Capture"):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        self.writer = cv2.VideoWriter(f'{file_name}.mp4',
                                      self.fourcc, 
                                      self.fps, 
                                      (frame_width,frame_height))
        self.frame = np.zeros((self.frame_height,self.frame_width,3), np.uint8)
        self.last_frame = np.zeros((self.frame_height,self.frame_width,3), np.uint8)
        self.stopped = False
    
    def start(self):
        Thread(target=self._write_frame, args=()).start()
        return self

    def _write_frame(self):
        if self.stopped:
            self._release_writer()
        while not self.stopped:
            # if not np.array_equal(self.frame, self.last_frame):
            self.writer.write(self.frame)
            self.last_frame = self.frame


    def _release_writer(self):
        self.writer.release()
    
    def stop(self):
        self.stopped = True


class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.

    source: https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self._show, args=()).start()
        return self

    def _show(self):
        while not self.stopped:
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True
        cv2.destroyAllWindows()

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True