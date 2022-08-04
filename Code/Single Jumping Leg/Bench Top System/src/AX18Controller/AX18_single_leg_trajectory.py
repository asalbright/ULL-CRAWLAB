
##########################################################################################
# AX18_single_leg_trajectory.py
#
# Script to move single leg, two dynamixel AX-18 servo end effector along an array-based trajectory
# Based on AX18Controller.py made by Eve Dang
#
# dynamixel_sdk module from: https://github.com/ROBOTIS-GIT/DynamixelSDK/tree/master/python/src/dynamixel_sdk
#
# AX-18 Info: https://emanual.robotis.com/docs/en/dxl/ax/ax-18a/#moving-speed
#
# Created: 11/3/2021 Eve Dang	
#
# Note: Change Homing Offset in Decimal in Extended Position control mode in Dynamixel Wizard 2.0.
##########################################################################################
import numpy as np
import datetime
from pathlib import Path    #For creating path files
import os
if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
from src.dynamixel_sdk import *
from AX18Controller import AX18, MultiAX18
import time 
from AX18_single_leg_Inverse import InverKinematic


# # First position
# First_position=[512,820]
Positions_Lists =[[512,820],[385,527],[512,820],[626,904],[512,820],[385,527],[512,820],[626,904],[512,820],[385,527],[512,820],[626,904],[512,820]]
# Positions_Lists.append(First_position)  

# # pX is xdistance between servo1 and end effector. 
# # pY is ydistance between servo1 and end effector.
# # L1 is length of link 1 between servo1 and servo2.
# # L2 is length of link 2 between servo2 and end effector. 

# # Sample dimensions
# # L1=6.5
# # L2=12
# # # Sample lists of pX, pY
# # X=[1,2,3,4,5]
# # Y=[1,2,3,4,5]
# # for i in range(len(X)):
# #     Positions_Lists.append(InverKinematic(X[i],Y[i],L1,L2))

PORT='/dev/tty.usbmodem141201'
ID_list=[1,2]
Actuator=MultiAX18(DEVICENAME=PORT,DXL_IDs=ID_list)
# Actuator.torque_enable()
# Actuator.move_to_position([500,500])
# # Position range:
# # ID1: 195-------988----original(150):512
# # ID2: 192-------901----original(240):820

def moveToPosition(PositionsList, start_time):
 
    readTimes = []
    servo1Positions = []
    servo2Positions = []
    servo1Speeds=[]
    servo2Speeds=[]
   
    CurrentVelocity=Actuator.move_to_position(PositionsList)
    readTimes.append(time.time() - start_time)
    CurrentPosition=Actuator.get_current_position()

    servo1Speeds.append(CurrentVelocity[0])
    servo2Speeds.append(CurrentVelocity[1])
     
    servo1Positions.append(CurrentPosition[0])
    servo2Positions.append(CurrentPosition[1])

    return (np.array((readTimes, servo1Positions, servo2Positions, servo1Speeds, servo2Speeds)).T)

move_data = np.zeros((1,5))
start_time = time.time()
for PositionsList in Positions_Lists:
    move_data = np.vstack((move_data, moveToPosition(PositionsList, start_time)))

Actuator.torque_disable()
move_data=move_data[1:]    
if __name__ == "__main__":
    save_path = Path.cwd()
    Experimental_Platform_path = save_path / 'Experimental Platform'
    Jumping_Data_path = Experimental_Platform_path / 'Jumping_Data'
    if not os.path.exists(Jumping_Data_path):
        os.makedirs(Jumping_Data_path)
    if not os.path.exists(Experimental_Platform_path):
        os.makedirs(Experimental_Platform_path)
   
data_filename = f'Jumping_Data_' + datetime.datetime.now().strftime('%H%M%S')+ '.csv'
path = Jumping_Data_path / data_filename

# np.savetxt(path, move_data, header='Time (s), Servo 1 Positions, Servo 2 Positions, Servo 1 Speeds, Servo 2 Speeds', delimiter=',')
Actuator.close()