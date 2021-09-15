################################################################################################################
# File Name: cube.py
# Author: Andrew Albright 
# 
# Description: file for loading in a cube for the robot to sit on
# Notes: https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24
################################################################################################################

import pybullet as p
import os

class Cube:
    def __init__(self, client):
        p.loadURDF(fileName="cube.urdf",
                   basePosition=[0, 0, -0.5],
                   useFixedBase=True,
                   physicsClientId=client)
