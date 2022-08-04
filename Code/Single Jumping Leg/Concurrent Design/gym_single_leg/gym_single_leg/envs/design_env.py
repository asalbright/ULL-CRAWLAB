#! /usr/bin/env python

###############################################################################
# design_env.py
#
#
# Created: 03/10/2022
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
#
# Notes:
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

from gym_single_leg.gym_single_leg.envs.control_env import SingleLegCtrlEnv

REWARD = {"H": 0, "E": 1}
CONTROL_MODE = {"PC": 0, "VC": 1}

logger = logging.getLogger(__name__)

class SingleLegDesEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 100
    }

    def __init__(self,
                 model,
                 robot_location,
                 reward_func=REWARD["H"],
                 ep_steps=1,
                 des_range=0.75,
                 sim_steps:int=240*2, 
                 sim_reward=REWARD["H"],
                 verbose=False):
       
        self.ctrl_model = model
        # Assign design attributes
        self.reward_func = reward_func
        self.des_range = des_range
        self.ep_steps = ep_steps
        self.timestep = 0

        # Assign the robot attributes
        self.robot_location = robot_location
        self.sim_steps = sim_steps
        self.sim_reward = reward
        self.client = self.open_eval_env()

        # Create action and observation spaces
        self.action_space = None
        self.observation_space = None
        self._create_box_spaces()

        # variable for logging height reached and power used
        self.height_reached = None
        self.power_used = None

        self.state = None
        self.done = False

        self.verbose = verbose

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
        if self.verbose:
            print(f"Testing Design Step {self.timestep}")

        self.timestep = self.timestep + 1
        
        # Create a dict of parms from the actions
        # Dict will look like {'kp_0': val, 'kd_0': val, 'kp_1': val, 'kd_1': val, ... 'kp_n': val, 'kd_n': val}
        it = iter(action)
        i = 0
        for _ in it:
            try:
                self.params[f'kv_{i}'] = i
                self.params[f'kd_{i}'] = next(it)
            except:
                self.params = {f'kv_{i}': i, f'kd_{i}': next(it)}
            i += 1

        # Pass the params to the eval env method
        self.eval_design(self.params)
        
        # Get the max height reached
        self.height_reached = np.max(self.state[0,:])

        # End the trial when we reach the maximum number of steps
        if self.timestep >= self.ep_steps:
            self.done = True

        # Get the reward depending on the reward function
        reward = self.calc_reward()

        return self.state, reward, self.done, {}

    def calc_reward(self):
        # ["Height", "Efficiency", "SpecHei", "SpHeEf"]
        # Get the height reached
        try:
            if self.reward_function == "Height": 
                reward = self.height_reached

            elif self.reward_function == "Efficiency":
                reward = self.height_reached / self.power_used

            elif self.reward_function == "SpecHei":
                error = abs(self.height_reached - self.specified_height) / self.specified_height
                reward = 1 / (error + 1)
                
        except:
            raise ValueError("REWARD FUNCTION NOT PROPERLY DEFINED PROPERLY")
            print("Proper reward functions are:" ,"\n",
                  "['Height', 'Efficiency', 'SpecHei', 'SpHeEf']")
            sys.exit()

        return reward

    def eval_design(self, params=None):
        # Open an eval env
        if self.client is None:
            self.client = self.open_eval_env()

        # Evaluate the agent
        obs = self.eval_env.reset()
        
        # Set the parameters if need be
        if not params is None:
            self.eval_env.leg.modify_design(params=params)

        done, state = False, None
        while not done:
            action, state = self.ctrl_model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = self.eval_env.step(action)
            # store state is self.state
            self.state[:,self.eval_env.timestep-1] = obs

        # Assign the power used
        self.power_used = self.eval_env.pogo_stick.get_power_used()
        # Close the self.eval_env
        self.eval_env.close()

    def open_eval_env(self):
        self.eval_env = SingleLegCtrlEnv(robot_location,
                                    ep_steps=self.sim_steps,
                                    control_mode=CONTROL_MODE["PC"],
                                    reward_type=REWARD["H"])
        return self.eval_env.client

    def close_eval_env(self):
        self.eval_env.close()
        self.client = None
        return self.client
                 
    def reset(self):
        """
        Reset the state of the system at the beginning of the episode.
        We set the rod position and velocity to zero, the actuator velocity to 
        zero, but let the actuator start at random locations along the length
        of the rod
        """ 
        # Reset the timestep, the jumping flag, the landings timestep, the power used list and the height reached list
        self.timestep = 0

        # Close the eval env
        self.client = self.close_eval_env()
        
        # Get the number of obs space params
        num_obs_params = self.observation_space.shape[0]
        self.state = np.zeros((num_obs_params, self.sim_steps))

        # Reset the power used and height reached attributes
        self.height_reached = None
        self.power_used = None

        # Reset the done flag
        self.done = False
        
        return np.array(self.state)

    def render(self, mode='human', close=False):
        '''
        Not applicable for this class
        '''
        pass

    def eval_env_info(self):
        """
        Return the info dict from the eval env
        """
        # Get the number of motors
        num_motors = len(self.eval_env.leg.motor_joints)
        # Get the motor max values
        motor_max_pos = self.eval_env.leg.max_motor_position
        motor_max_vel = self.eval_env.leg.max_motor_velocity
        # Get the number of flex joints
        num_flex_joints = len(self.eval_env.leg.flex_joints)
        # Get the flex joint nominal gain values
        pos_gain = self.eval_env.leg.position_gain
        vel_gain = self.eval_env.leg.velocity_gain

        return {"num_motors": num_motors, "motor_max_pos": motor_max_pos, "motor_max_vel": motor_max_vel,
                "num_flex_joints": num_flex_joints, "pos_gain": pos_gain, "vel_gain": vel_gain}

    def _create_box_spaces(self):
        leg_info = self.eval_env_info()
        # Get the number of flex joints in the system for action space
        num_flex_joints = leg_info["num_flex_joints"]
        # Get the gains
        pos_gain = leg_info["pos_gain"]
        vel_gain = leg_info["vel_gain"]
        # Get the number of motors in the system for action space
        num_motors = leg_info["num_motors"]
        # Get the max motor values
        max_motor_pos = leg_info["motor_max_pos"]
        max_motor_vel = leg_info["motor_max_vel"]

        # Create the action space
        min_pos_gain = pos_gain - pos_gain * self.des_range
        max_pos_gain = pos_gain + pos_gain * self.des_range
        min_vel_gain = vel_gain - vel_gain * self.des_range
        max_vel_gain = vel_gain + vel_gain * self.des_range

        # Create a lists: [kp_0, kv_0, kp_1, kv_1, ... , kp_n, kv_n] for n flex joints
        for _ in range(num_motors):
            try:
                low_limit = np.concatenate((low_limit, [min_pos_gain, min_vel_gain]))
                high_limit = np.concatenate((high_limit, [max_pos_gain, max_vel_gain]))
            except:
                low_limit = [min_pos_gain, min_vel_gain]
                high_limit = [max_pos_gain, max_vel_gain]

        self.action_space = spaces.Box(low=low_limit,
                                       high=high_limit,
                                       dtype=np.float32)
        
        # Create the observation space        
        obs_len = int(self.sim_steps)

        # Create a low list: [x_0[0...sim_steps], x_dot_0[0...sim_steps], x_1[0...sim_steps], x_dot_1[0...sim_steps], ... , x_n[0...sim_steps], x_dot_n[0...sim_steps]]
        for _ in range(num_motors):
            try:
                low_limit = np.concatenate((low_limit, [-max_motor_pos * np.ones(obs_len), -max_motor_vel * np.ones(obs_len)]))
                high_limit = np.concatenate((high_limit, [max_motor_pos * np.ones(obs_len), max_motor_vel * np.ones(obs_len)]))
            except:
                low_limit = [-max_motor_pos * np.ones(obs_len), -max_motor_vel * np.ones(obs_len)]
                high_limit = [max_motor_pos * np.ones(obs_len), max_motor_vel * np.ones(obs_len)]

        self.observation_space = spaces.Box(low=low_limit,
                                            high=high_limit,
                                            dtype=np.float32)        