#! /usr/bin/env python

###############################################################################
# poo_stick_jumping_contActionNonlinear.py
#
# Defines a pogo stick jumping environment for use with the openAI Gym.
# This version has a continuous range of inputs for the mass accel. input
#
# Created: 02/03/21
#   - Joshua Vaughan
#   - joshua.vaughan@louisiana.edu
#   - http://www.ucs.louisiana.edu/~jev9637
#
# Modified:
#   * 
#
# TODO:
#   * 02/23/21 - JEV 
#       - [ ] Choose reward function
#       - [ ] Determine proper actuator velocity and acceleration limits
#       - [ ] Decide if we need to use a more sophisticated solver
###############################################################################

import os
import sys
import gym
from gym import spaces
from gym.utils import seeding
import logging
import numpy as np
from scipy.integrate import solve_ivp
import datetime # for unique filenames
from pathlib import Path
from stable_baselines3.common.logger import record
from functions import rewardHeight, rewardHeightPunishPowerLinear, rewardHeightPunishPowerNonlinearCubed, rewardHeightPunishPowerNonlinearCubedSqrt, getReward


logger = logging.getLogger(__name__)


class PogoJumpingEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 100
    }

    def __init__(self, 
                 EVALUATING=False, 
                 MAX_STEPS=500, 
                 TAU=0.01, 
                 REWARD_FUNCTION='RewardHeight', 
                 OMEGA_X=0.99, 
                 SPRING_K=200000,
                 JUMP_TYPE="TimeJump", 
                 SAVE_DATA=False,
                 SAVE_NAME=None, 
                 SAVE_PATH=False):
        """
        Initialize with the parameters to match earlier papers from team at 
        Georgia Tech and Dr. Vaughan
        """
        self.gravity = 9.81              # accel. due to gravity (m/s^2)
        self.m_rod = 0.175               # mass of the pogo-stick rod (kg)
        self.m_act = 1.003               # mass of the pogo-stick rod (kg)
        self.mass = self.m_rod + self.m_act  # total mass
        self.f = 11.13                   # natural freq. (rad)
        self.wn = self.f * (2 * np.pi)   # Robot frequency (rad/s)
        self.zeta = 0.01                 # Robot damping ratio
        self.c = 2 * self.zeta * self.wn * self.mass  # Calculate damping coeff
        # self.k = self.mass * self.wn**2  # Calulate spring constant
        self.k = SPRING_K

        self.tau = TAU                   # seconds between state updates
        self.REWARD_FUNCTION = REWARD_FUNCTION  # variable for choosing what kind of reward function we want to use
        self.OMEGA_X = OMEGA_X           # value for weighting jump height vs power usage
        self.counter = 0                 # counter for number of steps
        
        self.MAX_STEPS = MAX_STEPS  # maximum number of steps to run
        self.JUMP_TYPE = JUMP_TYPE         # Flag to determine if episode terminates after "one" jump
        self.SAVE_DATA = SAVE_DATA          # set True to save episode data
        self.SAVE_PATH = SAVE_PATH       # path to save data to
        self.SAVE_NAME = SAVE_NAME
        self.EVALUATING = EVALUATING     # Flag if using the env for evaluating or training
        if self.EVALUATING:
            print('Environment is being used to Evaluate an agent.')
        # make the path for saving the data
        if self.SAVE_DATA:
            if not self.SAVE_PATH:
                save_path = Path.cwd()
                self.SAVE_PATH = save_path.joinpath('logs')
                if not os.path.exists(self.SAVE_PATH):
                    os.makedirs(self.SAVE_PATH)
            else:
                if not os.path.exists(self.SAVE_PATH):
                    os.makedirs(self.SAVE_PATH)

        self.jumping = False                  # Flag for defining if the system is jumping
        self.landed = 0                     # Counter for number of times the system has landed
        self.power_used = []
        self.height_reached = []

        # Define thesholds for trial limits
        # Either penalized heavily for exceeding these or end episode
        self.rod_max_position = np.inf                  # max jump height (m)
        self.rod_min_position = -0.125                  # amount the spring can compress by (m)
        self.rod_zero = -(self.mass * self.gravity / self.k)**(1/3) # amount the spring compresses by due to the weight of the system
        self.rod_max_velocity = np.inf                  # max rod velocity (m/s)
        self.act_max_position = 0.25                    # length of which actuator can move (m)
        self.act_min_position = 0.0                     # lower limit of the actuator (m)
        self.act_max_velocity = 1.0                     # max velocity of actuator (m/s)
        self.act_min_velocity = 0.0
        self.act_max_accel = 10.0                       # max acceleration of actuator (m/s^2)
        self.act_min_accel = 0.0

        # This action space is the range of acceleration mass on the rod
        self.action_space = spaces.Box(low=-self.act_max_accel,
                                       high=self.act_max_accel, 
                                       shape = (1,))
        
        high_limit = np.array([self.rod_max_position,      # max observable jump height
                               self.rod_max_velocity,      # max observable jump velocity
                               self.act_max_position,      # max observable actuator position
                               self.act_max_velocity])     # max observable actuator velocity

        low_limit = np.array([self.rod_min_position,       # max observable jump height
                              -self.rod_max_velocity,      # max observable jump velocity
                              self.act_min_position,       # max observable actuator position
                              -self.act_max_velocity])     # max observable actuator velocity
        
        self.observation_space = spaces.Box(high_limit, low_limit)

        self.seed()
        self.viewer = None
        self.state = None
        self.done = False
        self.x_act_accel = 0.0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """ 
        Take one step. Here, we just use a simple Euler integration of the ODE,
        clipping the input and states, if necessary, to meet system limits.
        
        We may need to replace the Euler integration with a more sophisticated
        method, given the nonlinearity of the system.
        """
        
        x, x_dot, x_act, x_act_dot = self.state
        self.counter = self.counter + 1
        
        # Get the action and clip it to the min/max trolley accel
        self.x_act_accel = np.clip(action[0], -self.act_max_accel, self.act_max_accel)
        
        # If the actuator is at a limit, don't let it accelerate in that direction
        if x_act >= self.act_max_position and self.x_act_accel > 0:
            self.x_act_accel = 0
        elif x_act <= self.act_min_position and self.x_act_accel < 0:
            self.x_act_accel = 0
        
        # Update the actuator states
        x_act_dot = x_act_dot + self.tau * self.x_act_accel

        # Keep velocity within limits
        x_act_dot = np.clip(x_act_dot, -self.act_max_velocity, self.act_max_velocity)
        
        x_act = x_act + self.tau * x_act_dot
        
        # Keep actuator position within limits
        x_act = np.clip(x_act, self.act_min_position, self.act_max_position)
        
        #determine if the system is in the air
        if x > 0:
            contact = 0
        else:
            contact = 1

        # Update the rod state, only allowing the spring and damper to act if
        # the rod is in contact with the ground.
        # TODO: A Albright - 03/30/2021 - consider changing k to a nonlinear value such as k**3 * x

        x_ddot = -contact * (self.k/self.mass * x**3 + self.c/self.mass * x_dot) - self.m_act/self.mass * self.x_act_accel - self.gravity
        x_dot = x_dot + self.tau * x_ddot
        x = x + self.tau * x_dot
        # x = np.clip(x, self.rod_min_position, 100)

        # set the return state
        self.state = (x, x_dot, x_act, x_act_dot)

        # determine if the pogo is jumping
        if x > 0 and self.jumping is False:
                self.jumping = True
        elif x <=0 and self.jumping is True:
                self.landed = self.landed + 1
                self.jumping = False

        # calculate the power used by the actuator
        power = float(abs(self.m_act * self.x_act_accel * x_act_dot))
        self.power_used.append(power)
        self.height_reached.append(float(x))

        # calculating and normalizing the reward:
        # TODO: 03/18/2021 - A Albright - Need to consider lowering the max jump height
        # also need to look at lowering how much the agent is punished for using 
        # power. 

        # set omega_p to be 1 - omega_x which was passed in
        omega_p = 1 - self.OMEGA_X
        power_min = self.mass * self.act_min_accel * self.act_min_velocity
        power_max = self.mass * self.act_max_accel * self.act_max_velocity
        position_min = 0
        position_max = 0.9

        # Get the reward depending on the reward function
        reward = getReward(omega_x=self.OMEGA_X, omega_p=omega_p, x=x, position_min=position_min, position_max=position_max, power=power, power_min=power_min, power_max=power_max, reward_function=self.REWARD_FUNCTION, power_used=self.power_used, height_reached=self.height_reached, counter=self.counter)

        # Define a boolean on whether we're exceeding limits or not. We can
        # penalize any of these conditions identically in the reward function
        # TODO: 03/16/2021 - Andrew Albright - Not using this at the moment
        space_limit =  x_act > self.act_max_position \
                or x_act < self.act_min_position \
                or x_act_dot > self.act_max_velocity \
                or x_act_dot < -self.act_max_velocity \

        # TODO: AA - 04/05/2021 - currently ending episode after a stutter jump
        # possibly consider ending with a single jump training to just compress once and jump. 
        # will need to think about not randomly initializing the actuator start position and instead starting it at max height

        # Determine if the episode needs to be terminated
        try:
            if self.JUMP_TYPE == "OneJump":
                if self.landed >= 1 or self.counter >= self.MAX_STEPS:
                    self.done = True
            elif self.JUMP_TYPE == "StutterJump":
                if self.landed >= 2 or self.counter >= self.MAX_STEPS:
                    self.done = True
            elif self.JUMP_TYPE == "TimeJump":
                if self.counter >= self.MAX_STEPS:
                    self.done = True
            else: raise Exception
        except Exception:
            print("JUMP TYPE NOT DEFINED PROPERLY")
            print()
            sys.exit()

        # append the data for current step to the episodes data array 
        time = self.tau * self.counter # time in seconds
        current_data = np.array([time, x, x_dot, x_act, x_act_dot, self.x_act_accel, power, self.OMEGA_X, 1-self.OMEGA_X, reward])
        self.episode_data[self.counter, :] = current_data

        # If the episode is finished, we create the csv of the episode data, including a text header.
        if self.done and self.SAVE_DATA:
            # If the data is only from OneJump or StutterJump and not an entire MAX_STEPs length, the save array needs to be resized
            if self.counter < self.MAX_STEPS:
                self.episode_data = self.episode_data[0:self.counter,:]
            
            header = 'Time, Rod Pos, Rod Vel, Actuator Pos, Actuator Vel, Actuator Accel, Power, Omega X, Omega P, Reward'
            data_filename = f'EpisodeData_{self.SAVE_NAME}.csv'
            data_path = self.SAVE_PATH / data_filename
            np.savetxt(data_path, self.episode_data, header=header, delimiter=',')

        return np.array(self.state), reward, self.done, {}

    def reset(self):
        """
        Reset the state of the system at the beginning of the episode.
        We set the rod position and velocity to zero, the actuator velocity to 
        zero, but let the actuator start at random locations along the length
        of the rod
        """ 
        # publish max height reached and power used data to tensorboard for tracking during training
        if len(self.height_reached):
            record('ep_max_height', max(self.height_reached))
            record('ep_power_used', sum(self.power_used))

        # initialize the starting state of the system
        # TODO: AA - 04/05/2021 - consider initializing the actuator to the highest position instead of the lowest postion
        # especially in the case where we want to jump only once. 
        if self.EVALUATING: 
            self.state = np.array([self.rod_zero,  # compressed sprint height above ground
                                   0,  # 0 initial vertical velocity
                                   (self.act_max_position)/2,  # 0 initial position of actuator
                                   0]) # 0 initial actuator velocity
        # during training randomly place the actuator along its path
        else:
            self.state = np.array([self.rod_zero,  # compressed spring height above ground
                                   0,  # 0 initial vertical velocity
                                   (self.act_max_position)/2,  # Start the actuator in the middle of its stroke
                                   0]) # 0 initial actuator velocity

        # Reset the counter, the jumping flag, the landings counter, the power used list and the height reached list
        self.counter = 0
        self.jumping = False
        self.landed = 0
        self.power_used = []
        self.height_reached = []
        
        # Reset the done flag
        self.done = False
        
        # If we are saving data, set up the array to save the data into until
        # we save it at the end of the episode
        self.episode_data = np.zeros((self.MAX_STEPS+1, 10)) #time, x, x_dot, x_act, x_act_dot, self.x_act_accel, self.power_used, reward
        self.episode_data[0,:] = np.array([0, self.state[0], self.state[1], self.state[2], self.state[3], self.x_act_accel, sum(self.power_used), self.OMEGA_X, 1-self.OMEGA_X, 0])

        return np.array(self.state)

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        # 600x400 because we're old school, but not 320x240 old.
        screen_width = 600
        screen_height = 400

        scale = 1.0 
        world_height = 3.0
        # scale = screen_width/world_width    # Scale according to width
        scale = screen_height/world_height    # Scale according to height
        
        # Define the pogo diameter and cable width in pixels
        rod_width = 10.0  # pixels
        rod_length= 4 * self.act_max_position * scale
        rod_yOffset = 25  # How far off the bottom of the screen is ground
        
        # Define the trolley size and its offset from the bottom of the screen (pixels)
        actuator_width = 20.0 
        actuator_height = 20.0

        x, x_dot, x_act, x_act_dot = self.state

        if self.viewer is None: # Initial scene setup
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # define the pogo rod as a polygon, so we can change its length later
            l,r,t,b = -screen_width/2, screen_width, rod_yOffset, -screen_height/2
            self.ground = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.ground.set_color(0.1, 0.1, 0.1)    # very dark gray
            self.viewer.add_geom(self.ground)
            
            # define the pogo rod as a polygon, so we can change its length later
            l,r,t,b = -rod_width/2, rod_width/2, rod_length/2, -rod_width/2
            self.rod = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.rodtrans = rendering.Transform(translation=(screen_width/2, rod_yOffset + x * scale))
            self.rod.add_attr(self.rodtrans)
            self.rod.set_color(0.25, 0.25, 0.25)    # dark gray
            self.viewer.add_geom(self.rod)
            
            # Define the actuator polygon
            l,r,t,b = -actuator_width/2, actuator_width/2, actuator_height/2, -actuator_height/2
            self.actuator = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.actuatortrans = rendering.Transform(translation=(screen_width/2, rod_yOffset + (x + x_act + self.act_max_position)*scale))
            self.actuator.add_attr(self.actuatortrans)
            self.actuator.set_color(0.85, 0.85, 0.85)    # light gray
            self.viewer.add_geom(self.actuator)


        # Move the rod
        self.rodtrans.set_translation(screen_width/2, rod_yOffset + x * scale)
        
        # move the trolley
        self.actuatortrans.set_translation(screen_width/2 , rod_yOffset + (x + x_act + self.act_max_position)*scale)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
        