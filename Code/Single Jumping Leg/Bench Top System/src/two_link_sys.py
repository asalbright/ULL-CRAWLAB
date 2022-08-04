################################################################################################################
# File Name: two_link_sys.py
# Author: Andrew Albright, a.albright1@louisiana.edu
# 
# Description: for controlling the two link benchtop system
# Notes: https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24
# TODO: ASA. June 08, 2021, Look into what the max force of the motors are
################################################################################################################

import os
import math
import numpy as np
import sys

JOINTS = {"motor": "motor", "flex": "flex", "rigid": "rigid", "slider": "slider"}
CONTROL_MODE = {"PC": 0, "VC": 1}

class SingleJumpingLeg:
    def __init__(self, 
                 max_motor_pos,
                 max_motor_vel, 
                 max_motor_force,
                 control_mode):

        self.client = client

        self.control_mode = control_mode

    def get_ids(self):
        return self.leg, self.client
    
    def apply_action(self, action):
        if self.control_mode == CONTROL_MODE["PC"]:
            # apply the action to the motors
            pass

        # If using velocity control mode
        elif self.control_mode == CONTROL_MODE["VC"]:
            # apply the action to the motors
            pass
    
    def get_observation(self):
        # Get the positions and velocities of the motors
        pass

    def get_height(self, link=None):
        # Using the camera tracking, get the height of the leg
        pass

    def contact(self, body:int) -> bool:
        # Use the camera to determine if the system is in contact with the ground
        pass

    def get_motor_states(self, joints=None):
        # used for get observation
        pass

    def get_camera_image(self):
        # Get the camera image
        pass