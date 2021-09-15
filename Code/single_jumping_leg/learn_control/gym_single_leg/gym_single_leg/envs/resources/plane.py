################################################################################################################
# File Name: plane.py
# Author: Andrew Albright 
# 
# Description: file for loading in a plane for the robot to sit on
# Notes: https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24
################################################################################################################

import pybullet as p
import os

class Plane:
    def __init__(self, client):
        f_name = os.path.join(os.path.dirname(__file__), 'simpleplane.urdf')
        p.loadURDF(fileName=f_name,
                   basePosition=[0, 0, 0],
                   physicsClientId=client)
