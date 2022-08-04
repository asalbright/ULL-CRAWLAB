###############################################################################
# two_link_flexible_env.py
#
#  
#
# Created: 11/28/2021
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
# Modified:
#
###############################################################################

import os
import sys
import gym
from datetime import datetime
from gym import spaces
from gym.utils import seeding
import logging
import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path

from gym_three_link_robot.gym_three_link_robot.envs.resources import ThreeLinkSerialManipulator
logger = logging.getLogger(__name__)

class ThreeLinkRobotArmEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 100
    }

    def __init__(self,
                 epSteps=300, # 10 seconds at tau = 0.01 
                 random_target=False,
                 random_start=False,
                 mod_design=False,
                 specifiedPosition=(4.2, 4.2),
                 captureData=False, 
                 saveDataName=None,
                 saveDataLocation=None):

        self.random_target = random_target              # generate a random target flag
        self.random_start = random_start                # generate a random start position flag
        self.mod_design = mod_design              # generate a random design flag

        self.timestep = 0                               # counter for number of steps
        self.init_norm_dist_to_target = None                 # initial distance to target
        self.target_range = 0.25                        # How close the ended effector must be to the target to be rewarded
        
        self.ep_steps = epSteps                         # maximum number of steps to run
        self.specified_pos = specifiedPosition          # height trying to jump to
        self.capture_data = captureData                 # set True to save episode data
        self.data_name = saveDataName                   # path to save data to
        if not saveDataLocation is None:
            self.data_location = Path(saveDataLocation)
        else:
            self.data_location = None

        # Reset the environment to get the action and observation space attributes
        self.reset()

        # Set the action space
        low_limit = np.array([-self.robot.tau_1_max,    # motor 1 torque
                              -self.robot.tau_2_max,     # motor 2 torque
                              -self.robot.tau_3_max])   # motor 3 torque
        high_limit = np.array([self.robot.tau_1_max,
                               self.robot.tau_2_max,
                               self.robot.tau_3_max])

        self.action_space = spaces.Box(low=low_limit, high=high_limit, dtype=np.float32)

        # Set the observation space
        # The observations are normalized
        low_limit = np.array([0,        # x distance to target
                              0,        # y distance to target
                              -1,       # motor 1 angle
                              -1,       # motor 1 velocity
                              -1,       # motor 2 angle
                              -1,       # motor 1 velocity
                              -1,       # motor 3 angle
                              -1])      # motor 1 velocity

        high_limit = np.array([1,
                               1,
                               1,
                               1,
                               1,
                               1,
                               1,
                               1])

        self.observation_space = spaces.Box(low=low_limit, high=high_limit, dtype=np.float32)

        # Set the seed
        self.seed()

        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, you are responsible for calling `reset()` to reset
        the environment's internal state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # Pass the action to the robot
        robot_state = self.robot.apply_action(action)
        # Set the state to the normalized robot state
        norm_robot_state = self.normalized_robot_state()

        # Get the x and y position of the end effector
        _, _, _, _, end_x, end_y = self.robot.get_link_end_xy()
        # Get the distance to the target
        x_dist_norm, y_dist_norm, _ = self.normalized_dist_to_target(end_x, end_y)

        # Set the state
        self.state = np.array([x_dist_norm, y_dist_norm] + [val for val in norm_robot_state])

        # Increment the timestep
        self.timestep += 1

        # Calculate the reward
        reward, self.done, info = self.calc_reward(end_x, end_y)

        # If the episode is done, reset the robot
        if self.timestep >= self.ep_steps:
            self.done = True

        if self.capture_data:
            self._capture_data(reward, end_x, end_y)

        return self.state, reward, self.done, info

    def normalized_robot_state(self, state=None, normr=[-1,1]):
        
        if state is None:
            state = self.robot.get_state()

        # Define the max robot state
        state_max = np.array([self.robot.theta1_max,
                              self.robot.theta1_dot_max,
                              self.robot.theta2_max,
                              self.robot.theta2_dot_max,
                              self.robot.theta3_max,
                              self.robot.theta3_dot_max])
        # Define the min robot state
        state_min = np.array([-self.robot.theta1_max,
                              -self.robot.theta1_dot_max,
                              -self.robot.theta2_max,
                              -self.robot.theta2_dot_max,
                              -self.robot.theta3_max,
                              -self.robot.theta3_dot_max])
        if normr is None:
            state_norm = (state - state_min) / (state_max - state_min)
        else:
            state_norm = normr[0] + ((state - state_min) * (normr[1] - normr[0])) / (state_max - state_min)
        
        return state_norm   

    def calc_reward(self, end_x, end_y):
        """ 
        Calculates the reward based on the error between the specified 
        endeffector position and the actual endeffector position

        Returns: 
            reward (float): negative distance between the specified position and the actual position
        """
        # Find the normalized distance between the end effector and the target
        _, _, norm_dist_to_target = self.normalized_dist_to_target(end_x, end_y)

        # Set the reward as the negative normalized distance
        reward = norm_dist_to_target * -1

        # Set up an initial dist to target to reward the agent if it reaches the target
        if self.init_norm_dist_to_target is None:
            self.init_norm_dist_to_target = norm_dist_to_target

        # Find the actual distance to the target
        _, _, dist_to_target = self.dist_to_target(end_x, end_y)
        # If the distance to the target is less than the target range, reward the agent
        if dist_to_target <= self.target_range:
            # Reward the agent for reaching the target the total sum of the negative rewards from its starting position if it never moved
            reward += np.abs(self.ep_steps * self.init_norm_dist_to_target)
            # Let the agent know that it has reached the target
            done = True
            info = {'target_reached': True}
        else:
            done = False
            info = {'target_reached': False}

        return reward, done, info
    
    def dist_to_target(self, end_x, end_y):
        """ 
        Calculates the distance between the specified endeffector position and the actual endeffector position

        Returns:
            x_dist, y_dist, dist_to_target (np.array): x and y distance from target, distance between the specified position and the actual position
        """ 
            
        x_dist = np.abs(end_x - self.specified_pos[0])
        y_dist = np.abs(end_y - self.specified_pos[1])
        dist_to_target = np.sqrt(x_dist**2 + y_dist**2)

        return np.array([x_dist, y_dist, dist_to_target])

    def normalized_dist_to_target(self, end_x, end_y, dist_to_target=None, normr=[0,1]):

        if dist_to_target is None:
            x_dist_to_target, y_dist_to_target, dist_to_target = self.dist_to_target(end_x, end_y)

        # Define the max distance to target
        dist_to_target_max = (self.robot.p[3] + self.robot.p[4] + self.robot.p[5]) * 2 # 2 is the radius the robot can create
        x_dist_to_target_max = dist_to_target_max
        y_dist_to_target_max = dist_to_target_max
        # Define the min distance to target
        dist_to_target_min = 0
        x_dist_to_target_min = dist_to_target_min
        y_dist_to_target_min = dist_to_target_min
        # Normalize the distance to target
        if normr is None:
            dist_to_target_norm = (dist_to_target - dist_to_target_min) / (dist_to_target_max - dist_to_target_min)
            x_dist_to_target_norm = (x_dist_to_target - x_dist_to_target_min) / (x_dist_to_target_max - x_dist_to_target_min)
            y_dist_to_target_norm = (y_dist_to_target - y_dist_to_target_min) / (y_dist_to_target_max - y_dist_to_target_min)
        else:
            dist_to_target_norm = normr[0] + ((dist_to_target - dist_to_target_min) * (normr[1] - normr[0])) / (dist_to_target_max - dist_to_target_min)
            x_dist_to_target_norm = normr[0] + ((x_dist_to_target - x_dist_to_target_min) * (normr[1] - normr[0])) / (x_dist_to_target_max - x_dist_to_target_min)
            y_dist_to_target_norm = normr[0] + ((y_dist_to_target - y_dist_to_target_min) * (normr[1] - normr[0])) / (y_dist_to_target_max - y_dist_to_target_min)

        return np.array([x_dist_to_target_norm, y_dist_to_target_norm, dist_to_target_norm])

    def reset(self):
        """
        Reset the state of the system at the beginning of the episode.
        We set the first link to 45deg and the second to 90deg, the actuator 
        velocities are set to zero.
        """ 
        # Set the robot
        self.robot = ThreeLinkSerialManipulator()
        # Reset the pogo_stick
        robot_state = self.robot.reset_state(random_start=self.random_start, mod_design=self.mod_design, params=None)
        # Normalize the robot state
        norm_robot_state = self.normalized_robot_state()

        # Get the distance to the target
        _, _, _, _, end_x, end_y = self.robot.get_link_end_xy()
        # Get the distance to the target
        x_dist_norm, y_dist_norm, _ = self.normalized_dist_to_target(end_x, end_y)

        # Set the state
        self.state = np.array([x_dist_norm, y_dist_norm] + [val for val in norm_robot_state])

        # Reset the counter
        self.timestep = 0
        # Reset the initial distance to target
        self.init_norm_dist_to_target = None
        # Reset the done flag
        self.done = False

        if self.random_target:
            x_pos = np.random.uniform(low=self.robot.workspace[0][0], high=self.robot.workspace[0][1])
            y_pos = np.random.uniform(low=self.robot.workspace[1][0], high=self.robot.workspace[1][1])
            self.specified_pos = (x_pos, y_pos)

        if self.capture_data:
            self._create_capture_data_array()

        return self.state

    def render(self, mode='human', close=False):

        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        
        # 600x400 because we're old school, but not 320x240 old.
        screen_width = 600
        screen_height = 400
        
        view_height = 2.1*(self.robot.p_nom[3] + self.robot.p_nom[4] + self.robot.p_nom[5]) # 3 times as wide as the robot links compined
        scale = screen_height/view_height

        link1_length = scale * self.robot.p[3]
        link2_length = scale * self.robot.p[4]
        link3_length = scale * self.robot.p[5]
        link1_width = scale * self.robot.p_nom[3] * 0.02 * self.robot.p[0]
        link2_width = scale * self.robot.p_nom[3] * 0.02 * self.robot.p[1]
        link3_width = scale * self.robot.p_nom[3] * 0.02 * self.robot.p[2]
        joint_radius = scale * self.robot.p_nom[3] * 0.1
        end_effector_width = scale * self.robot.p_nom[3] * 0.1


        x_zero = screen_width/2
        y_zero = screen_height/2

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Add polygon for the first link to the viewer
            l,r,t,b = 0, link1_length, link1_width/2, -link1_width/2
            link_1 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.link_1_transform = rendering.Transform()
            link_1.add_attr(self.link_1_transform)
            self.viewer.add_geom(link_1)

            # Add polygon for the second link to the viewer
            l,r,t,b = 0, link2_length, link2_width/2, -link2_width/2
            link_2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.link_2_transform = rendering.Transform()
            link_2.add_attr(self.link_2_transform)
            self.viewer.add_geom(link_2)

            # Add polygon for the third link to the viewer
            l,r,t,b = 0, link3_length, link3_width/2, -link3_width/2
            link_3 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.link_3_transform = rendering.Transform()
            link_3.add_attr(self.link_3_transform)
            self.viewer.add_geom(link_3)

            # Add a circle for the first joint to the viewer
            self.joint_1 = rendering.make_circle(joint_radius)
            self.joint_1.set_color(1, 0, 1)
            self.joint_1_transform = rendering.Transform()
            self.joint_1.add_attr(self.joint_1_transform)
            self.viewer.add_geom(self.joint_1)

            # Add a circle for the second joint to the viewer
            self.joint_2 = rendering.make_circle(joint_radius)
            self.joint_2.set_color(1, 0, 1)
            self.joint_2_transform = rendering.Transform()
            self.joint_2.add_attr(self.joint_2_transform)
            self.viewer.add_geom(self.joint_2)

            # Add a circle for the third joint to the viewer
            self.joint_3 = rendering.make_circle(joint_radius)
            self.joint_3.set_color(1, 0, 1)
            self.joint_3_transform = rendering.Transform()
            self.joint_3.add_attr(self.joint_3_transform)
            self.viewer.add_geom(self.joint_3)

            # Add a polygon for the end effector to the viewer
            l,r,t,b = end_effector_width*1.5, -end_effector_width*1.5, end_effector_width*1.5, -end_effector_width*1.5
            end_effector = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            end_effector.set_color(0, 0, 1)
            self.end_effector_transform = rendering.Transform()
            end_effector.add_attr(self.end_effector_transform)
            self.viewer.add_geom(end_effector)

            # Add a circle for the target to the viewer
            self.target = rendering.make_circle(joint_radius*0.5)
            self.target.set_color(1, 0, 0)
            self.target_transform = rendering.Transform()
            self.target.add_attr(self.target_transform)
            self.viewer.add_geom(self.target)

            # Add horizontal line for x-axis
            x_axis = rendering.Line((x_zero, y_zero), (screen_width, y_zero))
            self.viewer.add_geom(x_axis)

            # Add vertical line for y-axis
            y_axis = rendering.Line((x_zero, y_zero), (x_zero, screen_height))
            self.viewer.add_geom(y_axis)

            
        if self.state is not None:
            j1_x, j1_y, j2_x, j2_y, j3_x, j3_y = self.robot.get_joint_xy()
            joint1_x = x_zero + j1_x * scale
            joint1_y = y_zero + j1_y * scale
            joint2_x = x_zero + j2_x * scale
            joint2_y = y_zero + j2_y * scale
            joint3_x = x_zero + j3_x * scale
            joint3_y = y_zero + j3_y * scale
            _, _, _, _, end_effector_x, end_effector_y = self.robot.get_link_end_xy()
            end_effector_x = x_zero + end_effector_x * scale
            end_effector_y = y_zero + end_effector_y * scale
            target_x = x_zero + self.specified_pos[0] * scale
            target_y = y_zero + self.specified_pos[1] * scale
            
            self.link_1_transform.set_translation(joint1_x, joint1_y)
            self.link_2_transform.set_translation(joint2_x, joint2_y)
            self.link_3_transform.set_translation(joint3_x, joint3_y)
            self.joint_1_transform.set_translation(joint1_x, joint1_y)
            self.joint_2_transform.set_translation(joint2_x, joint2_y)
            self.joint_3_transform.set_translation(joint3_x, joint3_y)
            self.end_effector_transform.set_translation(end_effector_x, end_effector_y)
            self.target_transform.set_translation(target_x, target_y)
            
            self.link_1_transform.set_rotation(self.robot.state[0])
            self.link_2_transform.set_rotation(self.robot.state[0] + self.robot.state[2])
            self.link_3_transform.set_rotation(self.robot.state[0] + self.robot.state[2] + self.robot.state[4])
            self.end_effector_transform.set_rotation(self.robot.state[0] + self.robot.state[2] + self.robot.state[4])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: 
            self.viewer.close()
            self.viewer = None

    def _create_capture_data_array(self):
        self.ep_data = np.zeros((self.ep_steps+ 1, 4 + len(self.robot.state)))    # time, reward, end-effector x/y position, joint positions and velocities
        _, _, _, _, x_pos, y_pos = self.robot.get_link_end_xy()
        reward, _, _ = self.calc_reward(x_pos, y_pos)
        step_data = [0.0, reward, x_pos, y_pos] + list(self.robot.state.flatten())
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
            data_path = data_path / "Captured_Data"
            if not os.path.exists(data_path): 
                os.makedirs(data_path)
            self.temp_data_location = data_path
        # If the data path provided is not an existing directory, make it one
        elif not os.path.exists(self.data_location):
            data_path = Path.cwd()
            self.data_location = data_path / self.data_location
            os.makedirs(self.data_location)

    def _capture_data(self, reward, x_pos, y_pos):
        # append the data for current step to the episodes data array 
        time = self.robot.tau * self.timestep # time in seconds
        step_data = [time, reward, x_pos, y_pos] + list(self.robot.state.flatten())
        timestep_data = np.array(step_data)
        self.ep_data[self.timestep,:] = timestep_data

        # Save the data
        if self.done:
            # if the data does not fill the array declared for a full length episode
            if self.timestep < self.ep_steps:
                self.ep_data = self.ep_data[0:self.timestep,:]
            # Set the header and save the data
            header = 'Time, Reward, XPos, YPos, Joint1Pos, Joint1Vel, Joint2Pos, Joint2Vel, Joint3Pos, Joint3Vel'
            # Check if we are using the temp file location because one was not provided
            if self.data_name is None: data_name = self.temp_data_name
            else: data_name = self.data_name
            if self.data_location is None: data_location = self.temp_data_location
            else: data_location = self.data_location
            # Set the path and save the file
            data_path = data_location / f"{data_name}.csv"
            np.savetxt(data_path, self.ep_data, header=header, delimiter=',')