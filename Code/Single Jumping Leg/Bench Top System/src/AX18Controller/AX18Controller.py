##########################################################################################
# AX18Controller.py
#
# Script to control a single dynamixel AX-18 servo and multiple dynamixel AX-18 servos. 
#
# dynamixel_sdk module from: https://github.com/ROBOTIS-GIT/DynamixelSDK/tree/master/python/src/dynamixel_sdk
#
# AX-18 Info: https://emanual.robotis.com/docs/en/dxl/ax/ax-18a/
#
# Created: 11/3/2021 Eve Dang	
#
##########################################################################################

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
import time

class AX18:
    def __init__(self, DEVICENAME, DXL_ID,      
                       BAUDRATE=1000000,
                       DXL_MOVING_STATUS_THRESHOLD =5):
        
        self.DEVICENAME=DEVICENAME
        self.DXL_ID=DXL_ID
        self.ADDR_TORQUE_ENABLE=24
        self.BAUDRATE=BAUDRATE
        self.PROTOCOL_VERSION=1.0
        self.TORQUE_ENABLE=1
        self.TORQUE_DISABLE=0
        self.DXL_MOVING_STATUS_THRESHOLD=DXL_MOVING_STATUS_THRESHOLD

        
        #Velocity 
        self.ADDR_MOVING_VELOCITY=32
        self.LEN_MOVING_VELOCITY=2
        self.ADDR_PRESENT_VELOCITY=38
        self.LEN_PRESENT_VELOCITY=2
        self.DXL_VELOCITY_LIMIT=230
  
        #Position
        self.ADDR_GOAL_POSITION=30
        self.LEN_GOAL_POSITION=2
        self.ADDR_PRESENT_POSITION=36
        self.LEN_PRESENT_POSITION=2
        self.DXL_MINIMUM_POSITION_VALUE=0
        self.DXL_MAXIMUM_POSITION_VALUE=1023

        
        #Initialize PortHandler and PacketHandler instances
        self.portHandler=PortHandler(self.DEVICENAME)
        self.packetHandler=PacketHandler(self.PROTOCOL_VERSION)
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            print("Press any key to terminate...")
            getch()
            quit()
        if self.portHandler.setBaudRate(self.BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            print("Press any key to terminate...")
            getch()
            quit()

        # self.torque_enable()

    def reboot(self): 
        dxl_comm_result, dxl_error = self.packetHandler.reboot(self.portHandler, self.DXL_ID)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        
        print("[ID:%03d] reboot Succeeded\n" % self.DXL_ID)


    def torque_enable(self):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID,self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print(f"Dynamixel {self.DXL_ID} has been successfully connected")

    def get_current_velocity(self): 
        dxl_present_velocity, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, self.DXL_ID, self.ADDR_PRESENT_VELOCITY)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        
        print("[ID:%03d]  PresVel:%03d" % (self.DXL_ID, dxl_present_velocity))
        return dxl_present_velocity


    def set_velocity(self,GOAL_VELOCITY):
        if not GOAL_VELOCITY in range(self.DXL_VELOCITY_LIMIT):
            print("Invalid goal position!")
            print("Press any key to terminate...")
            getch()
            quit()
        else:
            dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, self.DXL_ID, self.ADDR_GOAL_VELOCITY, GOAL_VELOCITY)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            else:
                print(f"Velocity of Dynamixel {self.DXL_ID} has been changed successfully!")

    def get_current_position(self):
        dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, self.DXL_ID, self.ADDR_PRESENT_POSITION)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))

        print("[ID:%03d]  PresPos:%03d" % (self.DXL_ID, dxl_present_position))
        return dxl_present_position

    def move_to_position(self,GOAL_POSITION):
        dxl_goal_position_list = range(self.DXL_MINIMUM_POSITION_VALUE, self.DXL_MAXIMUM_POSITION_VALUE+1)
        if not GOAL_POSITION in dxl_goal_position_list:
            print("Invalid goal position!")
            print("Press any key to terminate...")
            getch()
            quit()
        else:
            #Read present position 
            dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, self.DXL_ID, self.ADDR_PRESENT_POSITION)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
                
            print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (self.DXL_ID, GOAL_POSITION, dxl_present_position))
            
            #Write goal position 
            while abs(GOAL_POSITION - dxl_present_position) > self.DXL_MOVING_STATUS_THRESHOLD:
                dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID, self.ADDR_GOAL_POSITION, GOAL_POSITION)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % self.packetHandler.getRxPacketError(dxl_error))
                    self.reboot()
                    break
                
            #Read present position
                dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, self.DXL_ID, self.ADDR_PRESENT_POSITION)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % self.packetHandler.getRxPacketError(dxl_error))
                
                print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (self.DXL_ID, GOAL_POSITION, dxl_present_position))
    
    
    def torque_disable(self):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))   
    
    def close(self):
        self.portHandler.closePort()

class SyncMultiAX18:
# Need to modify syncwrite function
# This class moves multiple servos at the same time 
    def __init__(self, DEVICENAME, DXL_IDs,      
                       BAUDRATE=1000000,
                       DXL_MOVING_STATUS_THRESHOLD =5):
        
        self.DEVICENAME=DEVICENAME
        self.DXL_IDs=DXL_IDs
        self.ADDR_TORQUE_ENABLE=24
        self.BAUDRATE=BAUDRATE
        self.PROTOCOL_VERSION=1.0
        self.TORQUE_ENABLE=1
        self.TORQUE_DISABLE=0
        self.DXL_MOVING_STATUS_THRESHOLD=DXL_MOVING_STATUS_THRESHOLD

        
        #Velocity 
        self.ADDR_MOVING_VELOCITY=32
        self.LEN_MOVING_VELOCITY=2
        self.ADDR_PRESENT_VELOCITY=38
        self.LEN_PRESENT_VELOCITY=2
        self.DXL_VELOCITY_LIMIT=230
  
        #Position
        self.ADDR_GOAL_POSITION=30
        self.LEN_GOAL_POSITION=2
        self.ADDR_PRESENT_POSITION=36
        self.LEN_PRESENT_POSITION=2
        self.DXL_MINIMUM_POSITION_VALUE=0
        self.DXL_MAXIMUM_POSITION_VALUE=1023
        
  
        #Initialize PortHandler and PacketHandler instances
        self.portHandler=PortHandler(self.DEVICENAME)
        self.packetHandler=PacketHandler(self.PROTOCOL_VERSION)
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            print("Press any key to terminate...")
            getch()
            quit()
        if self.portHandler.setBaudRate(self.BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            print("Press any key to terminate...")
            getch()
            quit()
        
        self.groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, self.ADDR_GOAL_POSITION, self.LEN_GOAL_POSITION)
        self.groupBulkRead = GroupBulkRead(self.portHandler, self.packetHandler) 
        for DXL_ID in self.DXL_IDs:
            dxl_addparam_result = self.groupBulkRead.addParam(DXL_ID, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupBulkRead addparam1 failed" % DXL_ID)
                quit()
            # dxl_addparam_result = self.groupBulkRead.addParam(DXL_ID, self.ADDR_PRESENT_VELOCITY, self.LEN_PRESENT_VELOCITY)
            # if dxl_addparam_result != True:
            #     print("[ID:%03d] groupBulkRead addparam2 failed" % DXL_ID)
            #     quit()


    def reboot(self): 
        for DXL_ID in self.DXL_IDs:
            dxl_comm_result, dxl_error = self.packetHandler.reboot(self.portHandler, DXL_ID)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            print("[ID:%03d] reboot Succeeded\n" % DXL_ID)


    def torque_enable(self):
        for DXL_ID in self.DXL_IDs:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s. AFTER REBOOT, RUN AGAIN!" % self.packetHandler.getRxPacketError(dxl_error))
                self.reboot()
            else:
                print(f"Dynamixel {DXL_ID} has been successfully connected")

    def get_current_velocity(self): 
        # current_velocity=[]
        # for DXL_ID in self.DXL_IDs:
        #     dxl_present_velocity = self.groupBulkRead.getData(DXL_ID, self.ADDR_PRESENT_VELOCITY, self.LEN_PRESENT_VELOCITY)
        #     print("[ID:%03d]  PresPos:%03d" % (DXL_ID, dxl_present_velocity))
        #     current_velocity.append(dxl_present_velocity) 
        # return current_velocity
        current_velocity=[]
        for DXL_ID in self.DXL_IDs:
            dxl_present_velocity, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, DXL_ID, self.ADDR_PRESENT_VELOCITY)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            
            print("[ID:%03d]  PresVel:%03d" % (DXL_ID, dxl_present_velocity))
            current_velocity.append(dxl_present_velocity)
        return current_velocity
    
    def torque_disable(self):
        for DXL_ID in self.DXL_IDs:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))   
            else:
                print(f"Dynamixel {DXL_ID} has been successfully disconnected")
    def close(self):
        self.groupSyncWrite.clearParam()
        self.groupBulkRead.clearParam()
        self.portHandler.closePort()

    def set_velocity(self,GOAL_VELOCITY_LIST):
        for GOAL_VELOCITY in GOAL_VELOCITY_LIST:
            if not GOAL_VELOCITY in range(self.DXL_VELOCITY_LIMIT):
                print("Invalid goal position!")
                print("Press any key to terminate...")
                getch()
                quit()
            else:
                dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_IDs[GOAL_VELOCITY_LIST.index(GOAL_VELOCITY)], self.ADDR_GOAL_VELOCITY, GOAL_VELOCITY)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % self.packetHandler.getRxPacketError(dxl_error))
                else:
                    print(f"Velocity of Dynamixel {self.DXL_IDs[GOAL_VELOCITY_LIST.index(GOAL_VELOCITY)]} has been changed successfully!")

    def get_current_position(self):
        current_position=[]
        for DXL_ID in self.DXL_IDs:
            dxl_present_position = self.groupBulkRead.getData(DXL_ID, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
            print("[ID:%03d]  PresPos:%03d" % (DXL_ID, dxl_present_position))
            current_position.append(dxl_present_position) 
        return current_position
        # for DXL_ID in self.DXL_IDs:
        #     dxl_addparam_result = self.groupSyncRead.addParam(DXL_ID)
        #     if dxl_addparam_result != True:
        #         print("[ID:%03d] groupSyncRead addparam failed" % DXL_ID)
        #         quit()     

        # dxl_comm_result = self.groupSyncRead.txRxPacket()
        # if dxl_comm_result != COMM_SUCCESS:
        #     print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        
        # current_position=[] #DEMO
        # for DXL_ID in self.DXL_IDs:
        # # Check if groupsyncread data of Dynamixel#1 is available
        #     dxl_getdata_result = self.groupSyncRead.isAvailable(DXL_ID, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
        #     if dxl_getdata_result != True:
        #         print("[ID:%03d] groupSyncRead getdata failed" % DXL_ID)
        #         quit()
        #     dxl_present_position = self.groupSyncRead.getData(DXL_ID, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
        #     print("[ID:%03d]  PresPos:%03d" % (DXL_ID, dxl_present_position))
        #     current_position.append(dxl_present_position) #DEMO
        
        # self.groupSyncRead.clearParam()
        # return current_position #DEMO


# Rewrite while loop to allow the servo have time to move and stop once it gets to goal position.
    def move_to_position(self,GOAL_POSITION_LIST):
        self.get_current_position()
        #CHECK GOAL POSITION
        dxl_goal_position_list = range(self.DXL_MINIMUM_POSITION_VALUE, self.DXL_MAXIMUM_POSITION_VALUE+1)
        for GOAL_POSITION in GOAL_POSITION_LIST:
            if not GOAL_POSITION in dxl_goal_position_list:
                print("Invalid goal position! Press any key to terminate...")
                getch()
                quit()
        for i in range(len(self.DXL_IDs)): 
                #Write goal position 
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(GOAL_POSITION_LIST[i])), DXL_HIBYTE(DXL_LOWORD(GOAL_POSITION_LIST[i])), DXL_LOBYTE(DXL_HIWORD(GOAL_POSITION_LIST[i])), DXL_HIBYTE(DXL_HIWORD(GOAL_POSITION_LIST[i]))]
            dxl_addparam_result = self.groupSyncWrite.addParam(self.DXL_IDs[i], param_goal_position)
            
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncWrite addparam failed" % (self.DXL_IDs[i]) )
                quit()
    
            dxl_comm_result = self.groupSyncWrite.txPacket()
            print('got it')
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            # Clear syncwrite parameter storage
            self.groupSyncWrite.clearParam()

class MultiAX18:
# This class moves multiple servos consecutively 
    def __init__(self, DEVICENAME, DXL_IDs):
        
        self.DEVICENAME=DEVICENAME
        self.DXL_IDs=DXL_IDs
        self.PROTOCOL_VERSION=1.0
        self.BAUDRATE=1000000

        self.SERVO=[]

        #Initialize PortHandler and PacketHandler instances
        self.portHandler=PortHandler(self.DEVICENAME)
        self.packetHandler=PacketHandler(self.PROTOCOL_VERSION)
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            print("Press any key to terminate...")
            getch()
            quit()
        if self.portHandler.setBaudRate(self.BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            print("Press any key to terminate...")
            getch()
            quit()
        for i in range(len(self.DXL_IDs)):
            self.SERVO.append(AX18(DEVICENAME=self.DEVICENAME, DXL_ID=self.DXL_IDs[i]))
        
    def reboot(self): 
        for i in range(len(self.DXL_IDs)):
            self.SERVO[i].reboot()

    def torque_enable(self):
        for i in range(len(self.DXL_IDs)):
            self.SERVO[i].torque_enable()
    
    def get_current_velocity(self): 
        current_velocity=[]
        for i in range(len(self.DXL_IDs)):
            current_velocity.append(self.SERVO[i].get_current_velocity())
        return current_velocity

    def torque_disable(self):
        for i in range(len(self.DXL_IDs)):
            self.SERVO[i].torque_disable()

    def close(self):
        self.portHandler.closePort()

    def set_velocity(self,GOAL_VELOCITY_LIST):
        for i in range(len(self.DXL_IDs)):
            self.SERVO[i].set_velocity(GOAL_VELOCITY_LIST[i])


    def get_current_position(self):
        current_position=[]
        for i in range(len(self.DXL_IDs)):
            current_position.append(self.SERVO[i].get_current_position())
        return current_position

    def move_to_position(self,GOAL_POSITION_LIST):
        current_velocity=[]
        for i in range(len(self.DXL_IDs)):
            self.SERVO[i].move_to_position(GOAL_POSITION=GOAL_POSITION_LIST[i])
            current_velocity.append(self.SERVO[i].get_current_velocity())
        return current_velocity
