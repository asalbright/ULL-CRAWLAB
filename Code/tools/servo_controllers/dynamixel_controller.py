from tools import dynamixel
import numpy as np
import sys

class Controller:
    def __init__(self, 
                 port=None, 
                 baudrate=None, 
                 timeout=None, 
                 servoIds=None):       
        # Establish a serial connection to the dynamixel network.
        # This usually requires a USB2Dynamixel
        self.serial = dynamixel.SerialStream(port=port, 
                                             baudrate=baudrate, 
                                             timeout=timeout)
        # Instantiate the dynamixel network object
        self.net = dynamixel.DynamixelNetwork(self.serial)
        # Populate our network with dynamixel objects
        self.servo_ids = servoIds
        for servo in self.servo_ids:
            newDynamixel = dynamixel.Dynamixel(servo, self.net)
            self.net._dynamixel_map[servo] = newDynamixel

        self.set_servo_ids()
        # Make sure we have at least one servo connected
        if not self.net.get_dynamixels():
            print('No Dynamixels Found!')
            sys.exit(0)
        else:
            # Servos were found
            print('Found some servos...')
            pass
        self.net.synchronize()
        
    def set_servo_ids(self):
        self.servos = list(range(max(self.servo_ids) + 1))
        self.servos[self.servo_ids[0]] = self.net.get_dynamixels()[0]
        self.servos[self.servo_ids[1]] = self.net.get_dynamixels()[1]
        self.net.synchronize()

    def close(self):
        self.serial.close()
        pass

    def setTarget(self, chan, target):
        actuator = self.servos[chan]
        actuator.goal_position = int(target)
        self.net.synchronize()

    def setSpeed(self, chan, speed):
        self.servos[chan].moving_speed = int(speed)
        self.net.synchronize()

    def setTorque(self, chan, torque):
        self.servos[chan].torque_enabled = True
        self.servos[chan].torque_limit = int(torque)
        self.servos[chan].max_torque = int(torque)
        self.net.synchronize()