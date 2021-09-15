################################################################################################################
# File Name: manual_control.py
# Author: Andrew Albright, a.albright1@louisiana.edu
# 
# Description: file is for loading in the leg and manually controlling it. Will be used for a demo and for
#              getting up to date on some sim-to-real processes
# Notes:
################################################################################################################

import gym
import numpy as np
import math
import pybullet as p
import pybullet_data
from pathlib import Path
from datetime import datetime
import time

import os
import sys
import pygame
from pygame.locals import *
import pybullet as p 
import pybullet_data
from tools import maestro_controller, dynamixel_controller

from single_leg.resources.single_jumping_leg_rigid import SingleJumpingLegRigid
from single_leg.resources.plane import Plane

# Main loop to run

def main():
    # Connect to the pybullet server
    client = connectToPybulletServer()

    # Load in bodies
    # planeId = p.loadURDF("plane.urdf", physicsClientId=client)
    motor_vel = 97 / 60 * 2 * np.pi     # rmp / sec * 2 * pi = rad/s
    leg = SingleJumpingLegRigid(client, maxMotorVel=motor_vel, 
                                        maxMotorPos=np.pi/6,
                                        maxMotorForce=1.8, 
                                        robotType="OLD_LEG",
                                        controlMode="POSITION_CONTROL")

    game_controller = pygameController()
    
    motor_input = [0.0, 0.0]
    
    # Create a parameter that is changable on screen to increase motor speed
    MotorSpeed = p.addUserDebugParameter('MotorSpeed', 0, 25, 10)
    ResetEnv = p.addUserDebugParameter("Reset", 1, 0, 0)
    # Find the max height to display on screen
    max_height = leg.get_height()
    location = [0.0, -0.50, max_height]
    height = f"Height: {max_height:.5f}"
    slider_location = p.addUserDebugText(height, location, textSize=2.5)

    while True:
        if p.readUserDebugParameter(ResetEnv):
            p.disconnect(physicsClientId=client)
            main()
        
        joy_input = game_controller.get_joystick()

        # Update the motor speed according to the slider on screen
        MotorSpeedInput = p.readUserDebugParameter(MotorSpeed)
        # leg.max_velocity = MotorSpeedInput

        # Update the max height text shown on screen
        height = leg.get_height()
        if height > max_height:
            max_height = height
            location = [0.0, -0.50, height]
            height = f"Height: {height:.5f}"
            slider_location = p.addUserDebugText(height, location, textSize=2.5, replaceItemUniqueId=slider_location)

        # Set the motor and servo velocities according to the controller inputs
        if joy_input[1] > 0.1:
            motor_input[0] = joy_input[1] * leg.max_position
        
        elif joy_input[1] < -0.1:    
            motor_input[0] = joy_input[1] * leg.max_position

        if joy_input[3] > 0.1:
            motor_input[1] = joy_input[3] * leg.max_position 
        
        elif joy_input[3] < -0.1:    
            motor_input[1] = joy_input[3] * leg.max_position 

        action = motor_input[0], motor_input[1]
        leg.apply_action(action)

        # Step the simulation
        p.stepSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)

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

if __name__ == "__main__":
    main()
