################################################################################################################
# File Name: single_jumping_leg.py
# Author: Andrew Albright, a.albright1@louisiana.edu
# 
# Description: file describing a two-link jumping leg both rigid and flexible
# Notes: https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24
# TODO: ASA. June 08, 2021, Look into what the max force of the motors are
################################################################################################################

import pybullet as p
import os
import math
import numpy as np
import sys

TRANSPARENCY = 1
RED = [1, 0, 0, TRANSPARENCY]
GREEN = [0, 1, 0, TRANSPARENCY]
BLUE = [0, 0.5, 1, TRANSPARENCY]
BLACK = [0, 0, 0, TRANSPARENCY]
PURPLE = [0.5, 0, 1, TRANSPARENCY]

JOINTS = {"motor": "motor", "flex": "flex", "rigid": "rigid", "slider": "slider"}
CONTROL_MODE = {"PC": 0, "VC": 1}

class SingleJumpingLeg:
    def __init__(self, 
                 client, 
                 robotType,
                 robotLocation,
                 maxMotorPos,
                 maxMotorVel, 
                 maxMotorForce,
                 controlMode):

        self.client = client

        # Set the leg type and load in the leg
        try:
            if robotType == "DEFAULT": 
                f_name = os.path.join(os.path.dirname(__file__), "Single_Leg_Jumping_Robot/Single_Leg_Jumping_Robot.urdf")
            elif robotType == "USER_SPECIFIED":
                if not os.path.exists(robotLocation):
                    raise Exception("Invalid robot path provided.")
                    sys.exit()
                    
                f_name = robotLocation
        except:
            print("Robot type not specified properly.")
            sys.exit()

        self.leg = p.loadURDF(fileName=f_name,
                              basePosition=[0.0, 0.0, 0.01],
                              useFixedBase=True,
                              physicsClientId=self.client,
                            #   flags=p.URDF_USE_SELF_COLLISION
                              )
                              
        # Motor dynamics
        self.max_motor_position = maxMotorPos
        self.max_motor_velocity = maxMotorVel
        self.max_motor_force = maxMotorForce

        # Set up the control mode variable
        self.control_mode = controlMode
        
        # Drag constant for slider joint
        self.c_sliding = 0.2    # based on a results of UHWM vs Steel

        # Set joint indices as found by p.getJointInfo() and set up joint/link parameters
        self.motor_joints = []
        self.sliding_joints = []
        self.flex_joints = []

        self._assign_joints()

        # Set the sliding joint(s) to be free sliding with some friction
        if len(self.sliding_joints) > 0:
            p.setJointMotorControl2(self.leg, 
                                    self.sliding_joints[0],
                                    p.VELOCITY_CONTROL,
                                    force=0.0,
                                    physicsClientId=self.client)

        self.height_printed = None                  # flag to print data in GUI

    def get_ids(self):
        return self.leg, self.client
    
    def apply_action(self, action):
        # Expects action to be two dimensional
        motor_control = list(action)

        # If using position control mode
        if self.control_mode == CONTROL_MODE["PC"]:
            # Clip the position angle to within the bounds
            motor_control = [np.clip(ii, -self.max_motor_position, self.max_motor_position) for ii in motor_control]
                                        
            for motor in range(len(motor_control)):
                p.setJointMotorControl2(self.leg,
                                        self.motor_joints[motor],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=motor_control[motor],
                                        targetVelocity=0,
                                        force=self.max_motor_force,
                                        maxVelocity=self.max_motor_velocity,
                                        physicsClientId=self.client)
            
        # If using velocity control mode
        elif self.control_mode == CONTROL_MODE["VC"]:
            # Clip the velocity command to within the bounds 
            motor_control = [np.clip(ii, -self.max_motor_velocity, self.max_motor_velocity) for ii in motor_control]

            # Check to see if the motors are at their limits
            joint_states = p.getJointStates(self.leg, 
                                            self.motor_joints, 
                                            physicsClientId=self.client)

            # Set the speed of the motors to be zero if the motors are at their limits and trying to move further
            for joint in range(len(joint_states)):
                if joint_states[joint][0] >= self.max_motor_position and motor_control[joint] > 0:
                    motor_control[joint] = 0
                elif joint_states[joint][0] <= -self.max_motor_position and motor_control[joint] < 0:
                    motor_control[joint] = 0

            # Apply the velocity control to the motors
            p.setJointMotorControlArray(self.leg, 
                                        self.motor_joints,
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=motor_control,
                                        forces = np.ones(len(self.motor_joints))*self.max_motor_force,
                                        physicsClientId=self.client)

    def get_observation(self):
        # Get the positions and velocities of the motors
        joint_states = p.getJointStates(self.leg, 
                                        self.motor_joints,
                                        physicsClientId=self.client)
        
        return_states = []
        # If there are joints to report on, i.e, if there are controlled joints
        if joint_states:
            for ii in range(len(joint_states)):
                for jj in range(2):     # 2 = position and velocity
                    return_states.append(joint_states[ii][jj])

        return np.array(return_states)

    def get_height(self, link=None):
        # TODO: ASA, June 07, 2021, Look into making this more dynamic to any robot. 
        if link is None: 
            link = self.sliding_joints[0]

        link_state = p.getLinkState(self.leg, 
                                    link, 
                                    physicsClientId=self.client)
        link_pos = link_state[0]    # cartesian coordinates
        link_height = link_pos[2]   # z coordinate
        return link_height

    def contact(self, body:int) -> bool:
        # Use the pybullet getContactPoints() method to check if the leg is touching body
        contact_points = p.getContactPoints(self.leg,
                                            body,
                                            physicsClientId=self.client)
        if len(contact_points) > 0:
            return True
        else:
            return False

    def get_link_state(self, link=None):
        # TODO: ASA, June 07, 2021, Look into making this more dynamic to any robot. 
        if link is None: 
            link = self.sliding_joints[0]

        link_state = p.getLinkState(self.leg, 
                                    link, 
                                    physicsClientId=self.client)
        return link_state

    def show_height(self):
        # TODO: ASA, June 07, 2021, Look into making this more dynamic to any robot. 
        if self.height_printed is None:
            slider_pos = p.getLinkState(self.leg, self.sliding_joints[0])[0]
            str_slider_pos = [ '%.2f' % elem for elem in slider_pos ]
            slider_location = p.addUserDebugText(f"{str_slider_pos}", slider_pos)
            self.height_printed = True
        else:
            slider_pos = p.getLinkState(self.leg, self.sliding_joints[0])[0]
            str_slider_pos = [ '%.2f' % elem for elem in slider_pos ]
            slider_location = p.addUserDebugText(f"{str_slider_pos}", slider_pos, replaceItemUniqueId=slider_location)
        pass

    def get_joint_states(self, joints=None):
        '''
        if joints is None:
        Returns tuple slider_joint_states, motor_joint_states, flex_joint_states
        else:
        Returns array (joint_states) for list of joints provided

        States returned: Position, Velocity, Reaction Forces, Motor Torque
        '''
        if joints is not None:
            joint_states = p.getJointStates(self.leg,
                                            joints,
                                            physicsClientId=self.client)
            return np.array(joint_states, dtype=object)

        else:
            slider_joint_states = p.getJointStates(self.leg,    
                                                   self.sliding_joints,
                                                   physicsClientId=self.client)
            motor_joint_states = p.getJointStates(self.leg,
                                                  self.motor_joints,
                                                  physicsClientId=self.client)
            flex_joint_states = p.getJointStates(self.leg, 
                                                 self.flex_joints,
                                                 physicsClientId=self.client)

            return slider_joint_states, motor_joint_states, flex_joint_states

    def _assign_joints(self):
        ''' 
        Separate the joints into sections for applying actions
        '''
        # Get the number of joints
        num = p.getNumJoints(self.leg, physicsClientId=self.client)

        joint_info = []
        for ii in range(num): 
            # Append the the joint info list the info of the ii'th joint
            joint_info.append(p.getJointInfo(self.leg, ii, physicsClientId=self.client))
            # create a variable to temporarily store the joint name
            joint_name = joint_info[ii][1].decode('utf-8')
            # If the ii'th joints name has "J_C" in it, it is a continuous joint
            if JOINTS["motor"] in joint_name:
                # Add the index of the motor joint to the motor joints list
                self.motor_joints.append(joint_info[ii][0])
                # Change the dynamics of the motor joint
                p.changeDynamics(self.leg, 
                                    joint_info[ii][0], 
                                    lateralFriction=self.c_sliding,
                                    maxJointVelocity=self.max_motor_velocity,      # there is an issue with this we can set it during the control
                                    jointLowerLimit=-self.max_motor_position,
                                    jointUpperLimit=self.max_motor_position,
                                    jointLimitForce=self.max_motor_force,
                                    physicsClientId=self.client)
                # Enable the joint force/torque sensor so we can track the internal forces on the joint
                p.enableJointForceTorqueSensor(self.leg,
                                                joint_info[ii][0],
                                                enableSensor=True,
                                                physicsClientId=self.client)
                # Change the color of the parent and child link
                p.changeVisualShape(self.leg,
                                    ii,
                                    rgbaColor=BLACK,
                                    physicsClientId=self.client)

            # If the ii'th joints name has a "slider" in it, it is sliding joint
            elif JOINTS["slider"] in joint_name:
                # Add the sliding joint to the slider joints list
                self.sliding_joints.append(joint_info[ii][0])
                # Change the dynamics of the slider joint
                p.changeDynamics(self.leg, 
                                    joint_info[ii][0], 
                                    lateralFriction=self.c_sliding,            # TODO: ASA, June 08, 2021, might need to change this with rolling friction
                                    physicsClientId=self.client)
                # Change the color of the child link
                p.changeVisualShape(self.leg,
                                    ii,
                                    rgbaColor=BLUE,
                                    physicsClientId=self.client)

            # If the ii'th joints name has a "flex" in it, it is a flex joint
            elif JOINTS["flex"] in joint_name:
                # Add the flex joint to the flex joints list
                self.flex_joints.append(joint_info[ii][0])
                # Change the dynamics of the flex joint
                # p.changeDynamics(self.leg, joint_info[ii][0])
                # Enable the joint force/torque sensor so we can track the internal forces on the joint
                p.enableJointForceTorqueSensor(self.leg,
                                                joint_info[ii][0],
                                                enableSensor=True,
                                                physicsClientId=self.client)
                # Change the color of the child and the parent
                p.changeVisualShape(self.leg,
                                    ii,
                                    rgbaColor=RED,
                                    physicsClientId=self.client)
                p.changeVisualShape(self.leg,
                                    ii-1,
                                    rgbaColor=RED,
                                    physicsClientId=self.client)


        # Change the color of the base link
        p.changeVisualShape(self.leg,
                            -1,
                            rgbaColor=PURPLE,
                            physicsClientId=self.client)


class SingleJumpingLegRigid(SingleJumpingLeg):
    def __init__(self, 
                 client, 
                 robotType,
                 robotLocation=None,
                 maxMotorPos=np.pi/4,
                 maxMotorVel=np.pi, 
                 maxMotorForce=100,
                 controlMode=CONTROL_MODE["PC"],):
        super().__init__(client, 
                         robotType,
                         robotLocation,
                         maxMotorPos,
                         maxMotorVel, 
                         maxMotorForce,
                         controlMode)

        # FIXME: ASA, July 23, 2021, line 264: JOINT_FIXED is what I think we need, but it is not working like I would expect
        for ii in self.flex_joints:     
            joint_info = p.getJointInfo(self.leg,
                                              ii,
                                              physicsClientId=self.client)
            parent_frame_pos = np.array(joint_info[14])
            child_frame_pos = parent_frame_pos * -1
            p.createConstraint(parentBodyUniqueId=self.leg, 
                               parentLinkIndex=ii-1,
                               childBodyUniqueId=self.leg,
                               childLinkIndex=ii,
                               jointType=p.JOINT_FIXED,
                               jointAxis=[0, 0, 0],
                               parentFramePosition=parent_frame_pos,
                               childFramePosition=child_frame_pos,
                               physicsClientId=self.client)

    def apply_action(self, action):
        # Call the apply action from the parent class
        super().apply_action(action)


class SingleJumpingLegFlexible(SingleJumpingLeg):
    def __init__(self, 
                 client, 
                 robotType,
                 robotLocation=None,
                 maxMotorPos=np.deg2rad(45),
                 maxMotorVel=np.deg2rad(90), 
                 maxMotorForce=0.01,
                 positionGain=12.75,
                 velocityGain=0.975,
                 maxFlexPos=np.deg2rad(15),
                 controlMode=CONTROL_MODE["PC"]):
        super().__init__(client, 
                         robotType,
                         robotLocation,
                         maxMotorPos,
                         maxMotorVel, 
                         maxMotorForce,
                         controlMode)
        
        self.position_gain = positionGain
        self.velocity_gain = velocityGain
        self.max_flex_position = maxFlexPos

    def apply_action(self, action):
        # Call the apply action from the parent class
        super().apply_action(action)
        
        # Get the joint angles of the flex joints
        joint_states = p.getJointStates(self.leg, 
                                        self.flex_joints, 
                                        physicsClientId=self.client)
        joint_angles = []
        
        forces = np.ones(len(self.flex_joints)) * self.max_flex_position * self.position_gain
        
        # Set the flex joints according to PD control using the passed gains
        p.setJointMotorControlArray(self.leg,
                                    self.flex_joints,
                                    p.POSITION_CONTROL,
                                    targetPositions=np.zeros(len(self.flex_joints)),
                                    targetVelocities=np.zeros(len(self.flex_joints)),
                                    forces=forces,
                                    positionGains=np.ones(len(self.flex_joints)) * self.position_gain,
                                    velocityGains=np.ones(len(self.flex_joints)) * self.velocity_gain,
                                    physicsClientId=self.client)
