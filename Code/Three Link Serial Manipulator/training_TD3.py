#!/usr/bin/env python

##########################################################
# training.py
# Author: Andrew Albright
# Date: 11/29/2021
#
# Description:
#   This file contains the training code for the
#   neural network.
##########################################################

import os
from pathlib import Path
import time
import numpy as np 
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass
import torch
import gym
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# from gym_three_link_robot.gym_three_link_robot.envs import ThreeLinkRobotArmEnv
import gym_three_link_robot.gym_three_link_robot

# Make sure a GPU is available for torch to utilize
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.is_available())

# Set Environment parameters
ENV_ID = "three_link_robot_arm-v0"
EP_STEPS = 300

# Set up the Training parameters
NUM_TRIALS = 5
N_TRAINING_STEPS = 1000000
SAVE_FREQ = 50000
ROLLOUT = 10000

# Set up the training seeds
NUM_CPUs = 5
INITIAL_SEED = 70504
EVALUATION_FREQ = 25000
RNG = np.random.default_rng(INITIAL_SEED)
TRIAL_SEEDS = np.abs(RNG.integers(low=0, high=10000, size=(NUM_TRIALS)))

# Set up header file name
FILE_NAME = "NonLimited_Random"
SAVE_PATH = Path.cwd() / FILE_NAME

def train_agents(seed=12345):
    # Set the seed to be an integer
    seed = int(seed)

    # Set up the training save paths
    data_name = f'agent_{seed}'
    models_path = SAVE_PATH.joinpath(f'trained_{data_name}')
    logs_path = SAVE_PATH.joinpath('logs')
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    # Set up the training environments5
    env = gym.make(ENV_ID)
    env.random_target = True

    # Set up the checkpoint callback to save models periodically
    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=models_path, name_prefix=f'{data_name}')

    # Set up the training model
    model = TD3(policy="MlpPolicy", 
                env=env, 
                learning_starts=ROLLOUT,
                verbose=1, 
                tensorboard_log=logs_path, 
                seed=seed)

    # model = TD3.load("Training_Data/TD3/trained_agent_1/models/agent_12345_150000_steps.zip", env=env)

    # open tensorboard with the following bash command: tensorboard --logdir ./logs/
    # train the agent
    model.learn(total_timesteps=N_TRAINING_STEPS, 
                tb_log_name=f'{data_name}', 
                callback=checkpoint_callback)
                
    # Save the model at the end                
    path = models_path / f'final_{data_name}'
    model.save(path=path)
    
    # close the environment
    env.close()

def run_multi_processing(function, i, n_processors):
    with Pool(processes=n_processors) as pool:
        return pool.map(function, i)

if __name__ == "__main__":
    start = time.time()
    run_multi_processing(train_agents, TRIAL_SEEDS, NUM_CPUs)
    end = time.time()
    total = end - start
    print(f'Total Time: {total}')
    # train_agents()
