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
from tools import dynamixel
# from tools.functions import connectToPybulletServer, pygameController, connectMotorController

from single_leg.resources.single_jumping_leg_rigid import SingleJumpingLegRigid
from single_leg.resources.plane import Plane

# Main loop to run
def main():
    # Settings
    BAUD_RATE = 1000000
    HIGHEST_SERVO_ID = 18
    SERIAL_PORT = 'COM5'
    SERVO_IDS = [16,18]

    # Establish a serial connection to the dynamixel network.
    # This usually requires a USB2Dynamixel
    serial = dynamixel.SerialStream(port = SERIAL_PORT,
                                    baudrate = BAUD_RATE,
                                    timeout = 5)

    # Instantiate the dynamixel network object
    net = dynamixel.DynamixelNetwork(serial)

    # Populate our network with dynamixel objects
    for servoId in SERVO_IDS:
        newDynamixel = dynamixel.Dynamixel(servoId, net)
        net._dynamixel_map[servoId] = newDynamixel

    # Make sure we have at least one servo connected
    if not net.get_dynamixels():
        print('No Dynamixels Found!')
        sys.exit(0)
    else:
        # Servos were found
        print('Found some servos...')
        pass

    # set actuator labels
    actuator1 = net.get_dynamixels()[0]
    actuator2 = net.get_dynamixels()[1]

    # Set the moving speed - 0-1023 can be used
    # For AX-12A the units are about 0.111rpm.
    actuator1.moving_speed = 300
    actuator2.moving_speed = 300

    # Set true to enable motion of the servos
    actuator1.torque_enable = True
    actuator2.torque_enable = True

    # maximum torque limit 0-1023 (0x3FF) 
    # Units are ~0.1%
    actuator1.torque_limit = 1000
    actuator2.torque_limit = 1000

    # Torque value of maximum output.- 0-1023 
    # Units are ~0.1%
    actuator1.max_torque = 1000
    actuator2.max_torque = 1000

    # Send all the commands to the servo to move to "start" location
    # Range of the values is 0-1023
    # Units are 0.29 degree
    servo_max_pos = int(30 / (300 / 1023)) # 102.3
    servo1_home_pos = 525
    servo2_home_pos = 325
    actuator1.goal_position = servo1_home_pos # start position (for newer servo leg)
    actuator2.goal_position = servo2_home_pos
    net.synchronize()

    motor_input = [0.0, 0.0]
    
    # pygame stuff
    pygame.init()
    pygame.joystick.init()
    joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
    for joystick in joysticks:
        print(joystick.get_name())

    clock = pygame.time.Clock()
        
    joy_input = [0.0, 0.0, 0.0, 0.0]
    done = False

    while not done:

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

        # Set the motor and servo velocities according to the controller inputs
        # Joystick 1
        if joy_input[1] > 0.05:
            target = -joy_input[1] * servo_max_pos + servo1_home_pos
            actuator1.goal_position = int(target)
        elif joy_input[1] < -0.05:    
            target = -joy_input[1] * servo_max_pos + servo1_home_pos
            actuator1.goal_position = int(target)
        else: 
            target = servo1_home_pos
            actuator1.goal_position = int(target)

        # Joystick 2
        if joy_input[3] > 0.05:
            target = -joy_input[3] * servo_max_pos + servo2_home_pos
            actuator2.goal_position = int(target)
        elif joy_input[3] < -0.05:    
            target = -joy_input[3] * servo_max_pos + servo2_home_pos
            actuator2.goal_position = int(target)
        else: 
            target = servo2_home_pos
            actuator2.goal_position = int(target)

        net.synchronize()
        clock.tick(60)

    # quit both the game controller and the motor controller
    game_controller.quit()

if __name__ == "__main__":
    main()
