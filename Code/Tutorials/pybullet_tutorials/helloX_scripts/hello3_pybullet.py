############################################################################
# File Name: hello3_pybullet.py
# Author: Andrew Albright 
# 
# Description: file for learning how to make and utilize a .urdf file
# Notes: https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24
############################################################################

import pybullet as p 
from time import sleep

p.connect(p.GUI)
p.loadURDF("simplecar.urdf") 
sleep(3)