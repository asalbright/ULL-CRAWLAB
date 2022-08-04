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
from pybullet_utils import bullet_client
import pybullet as p
import pybullet_data
import pkgutil
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import os
import cv2

from gym_single_leg.gym_single_leg.envs.resources.single_jumping_leg import SingleJumpingLegRigid, SingleJumpingLegFlexible
from tools.video_functions import VideoGet, VideoShow

REWARD = {"H": 0, "E": 1}
CONTROL_MODE = {"PC": 0, "VC": 1}
class SingleLegCtrlEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 robot_location,
                 show_GUI=False,
                 flexible=True,
                 ep_steps: int=240*2,
                 motor_args={"max_pos": np.deg2rad(30), "max_vel": 100, "max_force": 100},
                 flex_gains = {"kp": 1, "kd": 0.1},
                 max_flex_pos=np.deg2rad(15),
                 control_mode=CONTROL_MODE["PC"],
                 reward_type=REWARD["H"],
                 save_data=False,
                 save_info={"fname": "saved_data", "location": Path.cwd()}):
        
        # Connect to the server
        self.show_GUI = show_GUI
        self.client = self._connect_to_pybullet(self.show_GUI)

        # Create class attributes
        self.leg = None
        self.plane = None
        self.ep_steps = int(ep_steps)
        self.done = False
        self.timestep = 0
        self.jumping = False    
        self.num_jumps = -1     # The leg starts off the ground
        self.power_used = []
        self.total_reward = 0
        # Render attributes
        self.video_shower = None
        self.view_matrix = None
        # Solver analytics
        self.solver_analytics = None
        # Data capture attributes
        self.capture_data = save_data
        self.data_name = save_info["fname"]
        self.data_location = save_info["location"]
        # Load robot attributes
        self.robot_location = robot_location
        self.use_flexible_leg = flexible
        self.motor_max_pos = motor_args["max_pos"]                 # rad/s
        self.motor_max_vel = motor_args["max_vel"]                 # rad/s
        self.motor_max_force = motor_args["max_force"]             # N
        self.position_gain = flex_gains["kp"]
        self.velocity_gain = flex_gains["kd"]
        self.max_flex_position = max_flex_pos
        self.control_mode = control_mode
        # Reward attributes
        self.reward_case = reward_type

        # Reset the env
        self.reset()

        # Create the action / observation spaces for the gym env
        self.observation_space = None
        self.action_space = None
        self._create_box_spaces(self.control_mode)
        
    def step(self, action):
        # Feed action to the leg and get observation of legs's state
        self.leg.apply_action(action)
        self.solver_analytics = p.stepSimulation(physicsClientId=self.client)[0] # [0] because returns a list of a dictionary
        
        # Increase timesteps counter
        self.timestep = self.timestep + 1

        # Get the leg observaton
        leg_ob = self.leg.get_observation()

        # Get the height of the slider joint (used to represent the height of the leg)
        self.height = self.leg.get_height(link=self.leg.sliding_joints[0])

        # Is the leg on the ground
        self.grounded = self.leg.contact(self.plane)

        # Get the reward, see self.done conditions (line 127)
        reward = self.calc_reward(self.reward_case)

        # If the leg is off the ground and touches the ground
        if self.jumping:
            if self.grounded:
                self.num_jumps += 1
                self.jumping = False
        # If the leg is on the ground and jumps
        if not self.jumping:
            if not self.grounded:
                self.jumping = True

        # Done by reaching time limit
        if self.timestep >= self.ep_steps:
            self.done = True
            # If the leg never leaves the ground, punish it heavily
            reward = -1 * self.total_reward

        # Done by completting a jump
        elif self.num_jumps >= 1:
            self.done = True

        # If capture data is on, capture and save the data
        if self.capture_data:
            # Capture the data every timestep
            timestep_data = [self.timestep, reward, self.height]
            timestep_data.extend(leg_ob)

            self.ep_data[self.timestep,:] = timestep_data

            # Save the data
            if self.done:
                # if the data does not fill the array declared for a full length episode
                if self.timestep < self.ep_steps:
                    self.ep_data = self.ep_data[0:self.timestep,:]
                # Set the header and save the data
                header = 'Time, Reward, Height'
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

    def calc_reward(self, reward_case=REWARD["H"]):
        # Update the power used
        self._update_power_used()
        # Get the total power used
        total_power_used = self.get_power_used()

        if reward_case == REWARD["H"]:
            if self.grounded:
                reward =  self.height
            else:
                reward = 2 * self.height

        elif reward_case == REWARD["E"]:
            if self.grounded:
                reward = self.height / total_power_used    # efficiency = height / power
                 
            else:
                reward = 2 * (self.height / total_power_used)  # efficiency = height / power
        
        self.total_reward += reward
        return reward

    def _update_power_used(self):
        # Update the power used
        motor_joint_states = self.leg.get_joint_states(self.leg.motor_joints)  
        motor_joint_torques = motor_joint_states[:,3]
        motor_joint_velocities = motor_joint_states[:,1]
        self.power_used.append(motor_joint_torques * motor_joint_velocities)

    def get_power_used(self):
        return np.sum(self.power_used)

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
        self.plane = p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=self.client)

        # Load in either a rigid or flexible leg
        if self.use_flexible_leg: 
            self.leg = SingleJumpingLegFlexible(self.client, 
                                                robot_location=self.robot_location,
                                                max_motor_vel=self.motor_max_vel, 
                                                max_motor_pos=self.motor_max_pos,
                                                max_motor_force=self.motor_max_force,
                                                position_gain=self.position_gain,
                                                velocity_gain=self.velocity_gain,
                                                flex_max_pos=self.max_flex_position,
                                                control_mode=self.control_mode)
        else:
            self.leg = SingleJumpingLegRigid(self.client, 
                                             robot_location=self.robot_location,
                                             max_motor_vel=self.motor_max_vel, 
                                             max_motor_pos=self.motor_max_pos,
                                             max_motor_force=self.motor_max_force,
                                             control_mode=self.control_mode)

        # Reset done flags, counters and power used array
        self.done = False
        self.timestep = 0
        self.height = self.leg.get_height(link=self.leg.sliding_joints[0])
        if self.height > 0:
            self.jumping = True
            self.grounded = False
        else:
            self.jumping = False
            self.grounded = True
        self.num_jumps = -1     # The leg starts off the ground
        self.power_used = []
        self.total_reward = 0

        # Get observation to return
        leg_ob = self.leg.get_observation()

        # If capturing data is on, initialize the data capture array
        if self.capture_data:
             self._create_capture_data_array(leg_ob)

        return leg_ob

    # TODO: June 11, 2021, visualizing can be done using p.GUI but this might be necessary for saving a video.
    def render(self, mode='human'):
        # Get the latest sim frame
        frame = self.get_camera_image()
        # Start the video shower
        if self.video_shower is None:
            self.video_shower = VideoShow(frame).start()

        else:
            # Close the video shower
            if self.video_shower.stopped:
                self.video_shower.stop()
                self.video_shower = None
            # Update the video shower frame
            self.video_shower.frame = frame

    def get_camera_image(self, xres=640, yres=480):
        # Set up the view matrix
        # source: https://colab.research.google.com/drive/1u6j7JOqM05vUUjpVp5VNk0pd8q-vqGlx?authuser=1#scrollTo=7tbOVtFp1_5K
        camera_eye_position = [0.5, 0.0, 0.25]
        camera_target_position = [0.0, 0.0, 0.2]
        camera_up_vector = [0.0, 0.0, 1.0]
        self.view_matrix = p.computeViewMatrix(cameraEyePosition=camera_eye_position,
                                               cameraTargetPosition=camera_target_position,
                                               cameraUpVector=camera_up_vector,
                                               physicsClientId=self.client)

        # Set up the projection matrix
        near_plane = 0.01
        far_plane = 100
        fov = 60
        aspect = xres / yres
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near_plane, far_plane)

        width, height, rgb, _, _  = p.getCameraImage(xres, yres, self.view_matrix, self.projection_matrix, physicsClientId=self.client)
        np_img = np.reshape(rgb, (height, width, 4))
        frame = np_img[:, :, :3]

        return frame

    def _connect_to_pybullet(self, show_GUI):
        if show_GUI:
            self._p = bullet_client.BulletClient(connection_mode=p.GUI)
            self._p.resetDebugVisualizerCamera(cameraDistance=0.5, 
                                         cameraYaw=90.0, 
                                         cameraPitch=-15.0, 
                                         cameraTargetPosition=[0.0, 0.0, 0.0])
        else:
            self._p = bullet_client.BulletClient(connection_mode=p.DIRECT)
        self._p.resetSimulation()
        self._p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        # Set additional search paths for loading URDF files
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())

        client = self._p._client
        self._p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        return client

    def _create_box_spaces(self, control_mode):
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
        if control_mode == CONTROL_MODE["PC"]:
            self.action_space = gym.spaces.box.Box(
                low=-np.array(pos_space, dtype=np.float32),
                high=np.array(pos_space, dtype=np.float32))

        elif control_mode == CONTROL_MODE["VC"]:             
            self.action_space = gym.spaces.box.Box(
                low=-np.array(vel_space, dtype=np.float32),
                high=np.array(vel_space, dtype=np.float32))

        # Set up observation space parameters: motor positions a velocities
        self.observation_space = gym.spaces.box.Box(
            low=-np.array(obs_space, dtype=np.float32),
            high=np.array(obs_space, dtype=np.float32))
    
    def _create_capture_data_array(self, leg_ob):
        # Set up and set the first row of the capture data array
        self.ep_data = np.zeros((self.ep_steps + 1, 3 + 2 * len(self.leg.motor_joints)))         # (ep_steps) by (time, reward, height, motors position&velocity)
        # Get the reward
        rew = self.calc_reward(self.reward_case)
        # Assign the time, rew, and height
        step_data = [0, rew, self.height]
        # Extend the step data with the motor positions and velocities
        step_data.extend(leg_ob)
        # Assign the first row of the step data
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
        p.disconnect(physicsClientId=self.client)
        # If we have rendered in a separate thread, close the thread
        if not self.video_shower is None:
            self.video_shower.stop()
