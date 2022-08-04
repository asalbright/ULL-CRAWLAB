
##########################################################################################
# AX18_single_leg_Inverse.py
#
# Functions for single leg with two dynamixel AX-18 servos.
# Based on AX18Controller.py made by Eve Dang
#
# dynamixel_sdk module from: https://github.com/ROBOTIS-GIT/DynamixelSDK/tree/master/python/src/dynamixel_sdk
#
# MX-28 Info: https://emanual.robotis.com/docs/en/dxl/mx/mx-28/
#
# Created: 11/3/2021 Eve Dang	
#
##########################################################################################
import numpy as np

def InverKinematic(pX,pY,L1,L2,defaultAngle1=135,defaultAngle2=90):
    # pX is xdistance between servo1 and end effector. 
    # pY is ydistance between servo1 and end effector.
    # L1 is length of link 1 between servo1 and servo2.
    # L2 is length of link 2 between servo2 and end effector. 

    # solve for theta2:
    xInputTheta2 = (pX**2 + pY**2 - L1**2 - L2**2) / (2 * L1 * L2)
    yInputTheta2 = np.sqrt(1 - ((pX**2 + pY**2 - L1**2 - L2**2) / (2 * L1 * L2))**2)
    theta2 = np.arctan2((-1 * yInputTheta2),xInputTheta2)
    
    # solve for theta1:

    xInputTheta1 = L2 * np.cos(theta2) + L1
    yInputTheta1 = np.sqrt(pY**2 + pX**2 - (L2 * np.cos(theta2) + L1)**2)
    theta1 = np.arctan2(pY,pX) + np.arctan2(yInputTheta1,xInputTheta1)

    # convert angles from radians to servo inputs

    theta1 = 180 * (theta1 / np.pi) # convert from radians to degrees
    theta1 = theta1 - defaultAngle1 # take default angle into account
    theta1 = int(4095*theta1/360)  # convert from degrees to 0-1023
    if theta1 > 1808 or theta1 <-778:
        print("Angle 1 must be between -778 and 1808 degrees")
        theta1=0
    
    theta2 = 180 * (theta2 / np.pi) # convert from radians to degrees
    theta2 = theta2 - defaultAngle2 # take default angle into account
    theta2 = int(4095*theta2/360) # convert from degrees to 0-1023
    if theta2 > 2315 or theta2 <-274:
        print("Angle must be between -274 and 2315 degrees")
        theta2=0
    return [theta1,theta2]