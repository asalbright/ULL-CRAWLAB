import os
import sys
import glob
from pathlib import Path
from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter
from tkinter import filedialog
import cv2 


class VideoWriteThreaded():
    '''
    This class is used to write videos to a file in a seperate thread.
    '''

    def __init__(self, frame_width=640, frame_height=480, fps=60, file_name="Video_Capture"):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        self.writer = cv2.VideoWriter(f'{file_name}.avi',
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

class VideoShowThreaded():
    """
    Class that continuously shows a frame using a thread.

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

class VideoGetThreaded():
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

    def get_frame(self):
        return self.frame

    def stop(self):
        self.stopped = True