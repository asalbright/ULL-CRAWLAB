from pygame_controller import pygameController
from AX18Controller import MultiAX18
import time


if __name__ == "__main__":
    cont_input = pygameController()
    DEVICENAME = None
    DXL_IDs = None

    servo_controller = MultiAX18(DEVICENAME, DXL_IDs)

    while True:
        joy_input = cont_input.get_joystick()

        # Need to convert -1 to 1 to -rad to rad
        # Need to convert -rad to rad to servo_pos
        
        servo_controller.set_goal_position(joy_input)