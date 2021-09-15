############################################################################
# File Name: hello4_pybullet.py
# Author: Andrew Albright 
# 
# Description: file for learning how to make and utilize a .urdf file
# Notes: https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24
############################################################################

import pybullet as p
from time import sleep
from pathlib import Path

p.connect(p.GUI)
p.setGravity(0, 0, -10)
angle = p.addUserDebugParameter('Steering', -0.5, 0.5, 0)
throttle = p.addUserDebugParameter('Throttle', -20, 20, 0)
f_name = str(Path('Simple-Driving/simple_driving/resources/simplecar.urdf'))
car = p.loadURDF(f_name, [0, 0, 0.1])
f_name = str(Path('Simple-Driving/simple_driving/resources/simpleplane.urdf'))
plane = p.loadURDF(f_name)
sleep(3)

wheel_indices = [1, 3, 4, 5]
hinge_indices = [0, 2]

while True:
    user_angle = p.readUserDebugParameter(angle)
    user_throttle = p.readUserDebugParameter(throttle)
    for joint_index in wheel_indices:
        p.setJointMotorControl2(car, joint_index,
                                p.VELOCITY_CONTROL,
                                targetVelocity=user_throttle)
    for joint_index in hinge_indices:
        p.setJointMotorControl2(car, joint_index,
                                p.POSITION_CONTROL, 
                                targetPosition=user_angle)
    p.stepSimulation()