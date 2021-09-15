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

import pygame
from pygame.locals import *
import pybullet as p 
import pybullet_data
from tools import maestro_controller, dynamixel_controller


def getZipFileName(path=None):
    if path is None:
        print('Path not specified for Zip name finder.')
    
    else:
        files = glob.glob(str(path / '*.zip'))
        print(f'Number of Zip files found: {len(files)}')
        file_names = []
        for f in files:
            file_names.append(os.path.basename(f).split(sep='.')[0])
        print(f'File Names found: {file_names}')

    return file_names

def connectToPybulletServer():
    connection = p.connect(p.GUI)
    # Set the distance and angle of the camera on the target
    p.resetDebugVisualizerCamera(cameraDistance=0.70, 
                                    cameraYaw=90.0, 
                                    cameraPitch=-15.0, 
                                    cameraTargetPosition=[0.0, 0.0, 0.0])                           
    p.setGravity(0, 0, -5)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)

    return connection

def connectMotorController(servoType, port, maxPos, minPos, servoIds, baudrate=None, timeout=None):
    servo_type = servoType

    if servoType == "HITEC":
        controller = maestro_controller.Controller(ttyStr=port)

    elif servoType == "DYNAMIXEL":
        controller = dynamixel_controller.Controller(port=port, baudrate=baudrate, timeout=timeout, servoIds=servoIds)

    return controller

class pygameController():

    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
        for joystick in self.joysticks:
            print(joystick.get_name())
        
    def get_joystick(self):
        joy_input = [0.0, 0.0, 0.0, 0.0]
        for event in pygame.event.get():
            if event.type == JOYBUTTONDOWN:
                print(event)
            if event.type == JOYBUTTONUP:
                print(event)
            if event.type == JOYAXISMOTION:
                if event.axis == 0:
                    joy_input[0] = event.value
                if event.axis == 1:
                    joy_input[1] = event.value
                if event.axis == 2:
                    joy_input[2] = event.value
                if event.axis == 3:
                    joy_input[3] = event.value
                print(event)
            if event.type == JOYHATMOTION:
                print(event)
            if event.type == JOYDEVICEADDED:
                joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
                for joystick in joysticks:
                    print(joystick.get_name())
            if event.type == JOYDEVICEREMOVED:
                joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        return joy_input
    
    def quit(self):
        pygame.quit()
    



