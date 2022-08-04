###############################################################################
# training.py
#
# A script for training the pogo_sick env varying the weights on the reward
# 
#
# Created: 03/29/2021
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
#
###############################################################################

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
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor

from gym_pogo_stick.gym_pogo_stick.envs import PogoStickControlEnv
from custom_callbacks import * 
from stable_baselines3.common.callbacks import CallbackList

# Make sure a GPU is available for torch to utilize
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.is_available())

# Set Environment parameters
ENV_TYPE = "Nonlinear"
EP_STEPS = 400
# "One" , "Stutter" , "TimeJump"
JUMP_TYPE = "Stutter"   
# Determine the number of jumps to send to the env
JUMP_TYPES = {"Stutter": 2, "One": 1}

# Set up the Training parameters
NUM_TRIALS = 5
N_TRAINING_STEPS = 750000
ROLLOUT = int(N_TRAINING_STEPS * 0.05)
UPDATE_MECH_PARAMS = 1000
GAMMA = 0.99

# Set up the training seeds
INITIAL_SEED = 70504
RNG = np.random.default_rng(INITIAL_SEED)
TRIAL_SEEDS = RNG.integers(low=0, high=10000, size=(NUM_TRIALS))
NUM_CPUs = 5

# Define the Reward Function to be used
# "Height", "Efficiency"
REWARD_FUNCTIONS = [
                    "Height", 
                    "Efficiency", 
                    # "SpecHei", 
                    # "SpHeEf"
                    ]

def train_agents(seed=12345):

    for REWARD_FUNCTION in REWARD_FUNCTIONS:
        # Set up the training save paths
        data_name = f'{REWARD_FUNCTION[0:5]}_{JUMP_TYPE}'
        save_path = Path.cwd()
        save_path = save_path.joinpath(f'ctr_{data_name}')
        logs_path = save_path.joinpath('logs')
        models_path = save_path.joinpath('models')
        models_path = models_path.joinpath(f'{int(seed)}')
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if not os.path.exists(models_path):
            os.makedirs(models_path)

        # Set up training env
        env = PogoStickControlEnv(
                                  numJumps=JUMP_TYPES[JUMP_TYPE],
                                  linear=ENV_TYPE,
                                  modParams=True,
                                  epSteps=EP_STEPS, 
                                  evaluating=False,
                                  rewardType=REWARD_FUNCTION,
                                  )
        
        # set the trial seed for use during training
        trial_seed = int(seed)
        env.seed(seed=trial_seed)

        # wrap the env in modified monitor which plots to tensorboard the jumpheight
        env = Monitor(env)

        buffer_size = N_TRAINING_STEPS + 1
        model = TD3(policy="MlpPolicy", 
                    env=env, 
                    verbose=1, 
                    tensorboard_log=logs_path, 
                    buffer_size=buffer_size, 
                    learning_starts=ROLLOUT,
                    seed=trial_seed, 
                    gamma=GAMMA)

        # Set up callbacks
        ctr_log_cb = ControllerLogCallback()
        learn_design_cb = TrainingDesignContinuousCallback(train_freq=UPDATE_MECH_PARAMS, 
                                                         rew_func=REWARD_FUNCTION, 
                                                         sim_steps=EP_STEPS, 
                                                         data_name=data_name, 
                                                         model_path=models_path,
                                                         learn_steps=750,
                                                         verbose=True)
        callbacks = CallbackList([
                                  ctr_log_cb, 
                                  learn_design_cb
                                  ])

        # open tensorboard with the following bash command: tensorboard --logdir ./logs/
        # train the agent
        model.learn(total_timesteps=N_TRAINING_STEPS, 
                    callback=callbacks,
                    tb_log_name=f'ctr_{int(seed)}')
                    
        # Save the model at the end                
        path = models_path / f'{data_name}_final'
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