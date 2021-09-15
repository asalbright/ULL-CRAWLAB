###############################################################################
# pogo_stick_evaluation.py
#
# A script for evaluating the model produced by pogo_stick_training and saving
# the data 
#
# Created: 02/23/2021
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
#
###############################################################################
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import datetime
import csv
import re

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from functions import getZipFileName

import pogo_stick_jumping

# Declare folders for logging data
save_path = Path.cwd()
models_path = save_path.joinpath('models')
figures_data_path = save_path.joinpath('figures_data')

if not os.path.exists(models_path):
    os.makedirs(models_path)
if not os.path.exists(figures_data_path):
    os.makedirs(figures_data_path)

# Call function to get agent names from models_path folder
agent_names = getZipFileName(path=models_path)

# Define the environment ID's and set Environment parameters
env_id = 'pogo-stick-jumping-v01'       # Nonlinear Environment
MAX_EPISODE_LENGTH = 400
# "OneJump" , "StutterJump" , "TimeJump"
JUMP_TYPE = "StutterJump"
# "RewardHeightPunishPowerLinear" , "RewardHeightPunishPowerNonlinearCubed" , "RewardHeightPunishPowerNonlinearCubedSqrt" , "RewardHeightRewardEfficiency" , "RewardHeight"
REWARD_FUNCTION = "RewardHeight"

OMEGA_X_LOWER = 0.1
OMEGA_X_UPPER = 1
OMEGA_X_STEP = 0.1
OMEGA_X = np.arange(OMEGA_X_LOWER, OMEGA_X_UPPER, OMEGA_X_STEP)

for agent in agent_names:
    # make gym environment
    env = gym.make(env_id)
    # Set evaluating Flag to true
    env.EVALUATING = True
    # Set the max episodic length and the timestep spacing
    env.MAX_STEPS = MAX_EPISODE_LENGTH
    env.TAU = 0.01 
    # Set the env jump type
    env.JUMP_TYPE = JUMP_TYPE
    # save the data to the path
    env.SAVE_DATA = True
    env.SAVE_PATH = figures_data_path
    # Set the name of the .csv file
    env.SAVE_NAME = f'{agent}'
    env.REWARD_FUNCTION = REWARD_FUNCTION
    # set env.OMEGA_x
    OMEGA_X = re.split('[_]', agent)[1]
    OMEGA_X = int(OMEGA_X) * 0.01
    env.OMEGA_X = OMEGA_X
    # wrap the env in a Monitor and DummyVecEnv wrapper
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # load the agent from the models path
    model_id = agent

    model = TD3.load(path=models_path / model_id)

    # Evaluate the agent

    # Stable Baselines standard evaluation
    # TODO: 03/16/2021 - ASA - Currently not used
    #mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, render=True)
    #print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    done, state = False, False
    obs = env.reset()
    while not done:
        action, state = model.predict(obs, state=state, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
    env.close()
