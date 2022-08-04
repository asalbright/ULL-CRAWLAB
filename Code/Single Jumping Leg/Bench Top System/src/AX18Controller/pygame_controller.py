import pygame
from pygame.locals import *

class pygameController():

    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
        for joystick in self.joysticks:
            print(joystick.get_name())
        pass
        
    def get_joystick(self):
        joy_input = [0.0, 0.0, 0.0, 0.0]
        for event in pygame.event.get():
            if event.type == JOYBUTTONDOWN:
                print(event)
            if event.type == JOYBUTTONUP:
                print(event)
            if event.type == JOYAXISMOTION:
                if event.axis == 0:
                    joy_input[0] = event.value
                if event.axis == 1:
                    joy_input[1] = event.value
                if event.axis == 2:
                    joy_input[2] = event.value
                if event.axis == 3:
                    joy_input[3] = event.value
                print(event)
            if event.type == JOYHATMOTION:
                print(event)
            if event.type == JOYDEVICEADDED:
                joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
                for joystick in joysticks:
                    print(joystick.get_name())
            if event.type == JOYDEVICEREMOVED:
                joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        return joy_input
    
    def quit(self):
        pygame.quit()

if __name__ == "__main__":
    controller = pygameController()
    while True:
        joy_input = controller.get_joystick()
        
        print(joy_input)

    controller.quit()