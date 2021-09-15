################################################################################################################
# File Name: single_leg_env.py
# Author: Andrew Albright, a.albright1@louisiana.edu
# 
# Description: Reinforcement learning env based for the rigid two leg env
# Notes: https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24
#        This file could also be used to represent the flexible two leg when it is finished.
################################################################################################################

import gym
import numpy as np
from pybullet_utils import bullet_client as bc
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import os

from gym_single_leg.gym_single_leg.envs.resources.single_jumping_leg import SingleJumpingLegRigid, SingleJumpingLegFlexible
from gym_single_leg.gym_single_leg.envs.resources.plane import Plane
from gym_single_leg.gym_single_leg.envs.resources.cube import Cube

class SingleLegEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 robotType="DEFAULT",
                 robotLocation=None,
                 showGUI=False,
                 flexible=False,
                 epSteps: int=240*2,
                 maxMotorPos=np.deg2rad(30),
                 maxMotorVel=100,   # default for pybullet
                 maxMotorForce=100,
                 positionGain=1,
                 velocityGain=1,
                 maxFlexPosition=np.deg2rad(15),
                 controlMode="POSITION_CONTROL",
                 rewardCase="HEIGHT",
                 captureData=False,
                 saveDataName=None,
                 saveDataLocation=None):

        super(SingleLegEnv, self).__init__()
        
        # Connect to the server
        self.client = self._connect_to_pybullet(showGUI)

        # Create class description variables
        self.leg = None
        self.ep_steps = int(epSteps)
        self.done = False
        self.rendered_img = None
        self.view_matrix = None
        self.solver_analytics = None
        self.capture_data = captureData
        self.data_name = saveDataName
        self.data_location = saveDataLocation
        self.robot_type = robotType
        self.robot_location = robotLocation
        self.use_flexible_leg = flexible

        self.motor_max_pos = maxMotorPos                 # rad/s
        self.motor_max_vel = maxMotorVel                 # m/s
        self.motor_max_force = maxMotorForce
        self.position_gain = positionGain
        self.velocity_gain = velocityGain
        self.max_flex_position = maxFlexPosition
        self.control_mode = controlMode
        self.reward_case = rewardCase

        # Create class updating variables
        self.timestep = 0
        self.power_used = []
        
        # Set the distance and angle of the camera on the target
        p.resetDebugVisualizerCamera(cameraDistance=1.0, 
                                     cameraYaw=90.0, 
                                     cameraPitch=-15.0, 
                                     cameraTargetPosition=[0.0, 0.0, 0.0])

        # Reset the env
        self.reset()

        # Create the action / observation spaces for the gym env
        self.observation_space = None
        self.action_space = None
        self._create_box_spaces(controlMode)
        
        # Reduce length of episodes for RL algorithms
        # According to documentation, this should not be changed unless absolutely needed
        # p.setTimeStep(1/30, self.client)
        
    def step(self, action):
        # Feed action to the leg and get observation of legs's state
        self.leg.apply_action(action)
        self.solver_analytics = p.stepSimulation(physicsClientId=self.client)[0] # [0] because returns a list of a dictionary
        
        # Increase timesteps counter
        self.timestep = self.timestep + 1

        # Get the leg observaton
        leg_ob = self.leg.get_observation()

        # Calculate reward based on the height the slider reaches
        reward = self.calc_reward(self.reward_case)

        # Done by reaching time limit
        if self.timestep >= self.ep_steps:
            self.done = True
        # TODO: Done by jumping once, use contact points

        # If capture data is on, capture and save the data
        if self.capture_data:
            # Capture the data every timestep
            timestep_data = [self.timestep, reward]
            timestep_data.extend(leg_ob)

            self.ep_data[self.timestep,:] = timestep_data

            # Save the data
            if self.done:
                # if the data does not fill the array declared for a full length episode
                if self.timestep < self.ep_steps:
                    self.ep_data = self.ep_data[0:self.timestep,:]
                # Set the header and save the data
                header = 'Time, Reward'
                for ii in range(len(self.leg.motor_joints)):
                    header += f", Motor{ii+1}Pos, Motor{ii+1}Vel"
                # Check if we are using the temp file location because one was not provided
                if self.data_name is None: data_name = self.temp_data_name
                else: data_name = self.data_name
                if self.data_location is None: data_location = self.temp_data_location
                else: data_location = self.data_location
                # Set the path and save the file
                data_path = data_location / f"{data_name}.csv"
                np.savetxt(data_path, self.ep_data, header=header, delimiter=',')

        return leg_ob, reward, self.done, dict()

    def calc_reward(self, reward_case="HEIGHT"):
        '''
            power = force * velocity
        '''
        motor_joint_states = self.leg.get_joint_states(self.leg.motor_joints)  
        motor_joint_torques = motor_joint_states[:,3]
        motor_joint_velocities = motor_joint_states[:,1]
        self.power_used.append(motor_joint_torques * motor_joint_velocities)
        total_power_used = np.sum(self.power_used)
        height_reached = self.leg.get_height(link=self.leg.sliding_joints[0])

        if reward_case == "HEIGHT":
            return height_reached

        elif reward_case == "EFFICIENCY":
            efficiency = height_reached / total_power_used
            return efficiency 

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(numSolverIterations=1000,
                                    # numSubSteps=100,
                                    erp=1e-9,
                                    contactERP=1e-9,  
                                    frictionERP=1e-9,
                                    solverResidualThreshold=1e-9, 
                                    contactSlop=0.0005,
                                    reportSolverAnalytics=True,
                                    physicsClientId=self.client)
                                    # 08/04/21 - JEV - I think the issue is the ODE solver. 
                                    # I adjusted some of its parameters here, which seemed to help.

        # Load in the base plane
        # Plane(self.client)
        Cube(self.client)

        # Load in either a rigid or flexible leg
        if self.use_flexible_leg: 
            self.leg = SingleJumpingLegFlexible(self.client, 
                                                robotType=self.robot_type,
                                                robotLocation=self.robot_location,
                                                maxMotorVel=self.motor_max_vel, 
                                                maxMotorPos=self.motor_max_pos,
                                                maxMotorForce=self.motor_max_force,
                                                positionGain=self.position_gain,
                                                velocityGain=self.velocity_gain,
                                                maxFlexPos=self.max_flex_position,
                                                controlMode=self.control_mode)
        else:
            self.leg = SingleJumpingLegRigid(self.client, 
                                             robotType=self.robot_type,
                                             robotLocation=self.robot_location,
                                             maxMotorVel=self.motor_max_vel, 
                                             maxMotorPos=self.motor_max_pos,
                                             maxMotorForce=self.motor_max_force,
                                             controlMode=self.control_mode)

        # Reset done flag, timestep counter and power used tracker
        self.done = False
        self.timestep = 0
        self.power_used = []

        # Get observation to return
        leg_ob = self.leg.get_observation()

        # If capturing data is on, initialize the data capture array
        if self.capture_data:
             self._create_capture_data_array(leg_ob)

        return leg_ob

    # TODO: June 11, 2021, visualizing can be done using p.GUI but this might be necessary for saving a video.
    def render(self, mode='human'):
        render_res = 500
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((render_res, render_res, 4)))

        camera_target = [0.0, 0.0, 0.0]
        camera_dist = 0.1
        camera_yaw = 0
        camera_pitch = 0
        camera_roll = 0
        camera_up = 2
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(camera_target,
                                                               camera_dist,
                                                               camera_yaw,
                                                               camera_pitch,
                                                               camera_roll,
                                                               camera_up)

        frame = p.getCameraImage(render_res, render_res, self.view_matrix)[2]
        frame = np.reshape(frame, (render_res, render_res, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)

    def _connect_to_pybullet(self, showGUI):
        if showGUI:
            client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        else: 
            client = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)

        return client

    def _create_box_spaces(self, controlMode):
        '''
        Creates continuous action and observation spaces which are symmetrical in regards to max position or velocity

        action space: [-max_pos/vel, max_pos/vel] for n motors (note position OR vel not pos divided vel)
        observation space: [-[max_pos, max_vel], [max_pos, max_vel]]
        '''
        pos_space = []
        vel_space = []
        obs_space = []
        for ii in range(len(self.leg.motor_joints)):
            pos_space.append(self.motor_max_pos)
            vel_space.append(self.motor_max_vel)
            obs_space.extend([self.motor_max_pos, self.motor_max_vel])
            
        # Set up action space parameters
        if controlMode == "POSITION_CONTROL":
            self.action_space = gym.spaces.box.Box(
                low=-np.array(pos_space, dtype=np.float32),
                high=np.array(pos_space, dtype=np.float32))

        elif controlMode == "VELOCITY_CONTROL":             
            self.action_space = gym.spaces.box.Box(
                low=-np.array(vel_space, dtype=np.float32),
                high=np.array(vel_space, dtype=np.float32))

        # Set up observation space parameters: motor positions a velocities
        self.observation_space = gym.spaces.box.Box(
            low=-np.array(obs_space, dtype=np.float32),
            high=np.array(obs_space, dtype=np.float32))
    
    def _create_capture_data_array(self, leg_ob):
        # Set up and set the first row of the capture data array
        self.ep_data = np.zeros((self.ep_steps + 1, 2 + 2 * len(self.leg.motor_joints)))         # time, slider height, motors position&velocity
        step_data = [0, self.leg.get_height()]
        step_data.extend(leg_ob)
        self.ep_data[0,:] = step_data
        # If the data name is blank create a temp name to save the data under
        if self.data_name is None: 
            self.temp_data_name = f"Data_{datetime.now().strftime('%m%d%Y_%H%M%S')}"
        # If the date in not appended to the end of the name append it
        elif not datetime.now().strftime('%m%d%Y') in self.data_name:
            self.temp_data_name = self.data_name
            self.data_name = f"{self.temp_data_name}_{datetime.now().strftime('%m%d%Y_%H%M%S')}"
        # If the date is appended to the end, update the time
        else:
            self.data_name = f"{self.temp_data_name}_{datetime.now().strftime('%m%d%Y_%H%M%S')}"
            
        # If the data location is blank, create a temp data location to save to
        if self.data_location is None: 
            data_path = Path.cwd()
            data_path = data_path.joinpath("Captured_Data")
            if not os.path.exists(data_path): 
                os.makedirs(data_path)
            self.temp_data_location = data_path
        # If the data path provided is not an existing directory, make it one
        elif not os.path.exists(self.data_location):
            os.makedirs(self.data_location)

    def close(self):
        p.disconnect(self.client)
