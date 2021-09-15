###############################################################################
# training_linear.py
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
import datetime
import time
import numpy as np 
import matplotlib.pyplot as plt
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass
import gym
import torch
import stable_baselines3
from stable_baselines3 import PPO, TD3
from custom_callbacks import EvalCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise

import pogo_stick_jumping

# Make sure a GPU is available for torch to utilize
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.is_available())

# Define the environment ID's and set Environment parameters
env_id = 'pogo-stick-jumping-v01'       # Nonlinear Environment
eval_env_id = 'pogo-stick-jumping-v01'
MAX_EPISODE_LENGTH = 400
# "OneJump" , "StutterJump" , "TimeJump"
JUMP_TYPE = "OneJump"   

# Set up the Training parameters
NUM_TRIALS = 10
N_TRAINING_STEPS = 5000
ROLLOUT = 5000
GAMMA = 0.99

# Set up the training seeds
INITIAL_SEED = 70504
EVALUATION_FREQ = 10000
RNG = np.random.default_rng(INITIAL_SEED)
TRIAL_SEEDS = RNG.integers(low=0, high=1000, size=(NUM_TRIALS))

# Define the Reward Function to be used
# "RewardHeightPunishPowerLinear" , "RewardHeightPunishPowerNonlinearCubed" , "RewardHeightPunishPowerNonlinearCubedSqrt" , "RewardHeightRewardEfficiency"
REWARD_FUNCTION = "RewardHeightRewardEfficiency"

# Set up the training save paths
data_name = f'Eff{JUMP_TYPE}'
save_path = Path.cwd()
save_path = save_path.joinpath(f'train_{data_name}')
logs_path = save_path.joinpath('logs')
models_path = save_path.joinpath('models')
if not os.path.exists(logs_path):
    os.makedirs(logs_path)
if not os.path.exists(models_path):
    os.makedirs(models_path)

# Begin loop for testing different Omega_X values
def trainAgents(seed):
        
    # Set up training env
    env = gym.make(env_id)
    env.MAX_STEPS = MAX_EPISODE_LENGTH
    env.REWARD_FUNCTION = REWARD_FUNCTION
    env.JUMP_TYPE = JUMP_TYPE

    # Set up evaluation env (used to generate "Best Agents")
    eval_env = gym.make(eval_env_id)
    eval_env.MAX_STEPS = MAX_EPISODE_LENGTH
    eval_env.EVALUATING = True
    eval_env.REWARD_FUNCTION = REWARD_FUNCTION
    eval_env.JUMP_TYPE = JUMP_TYPE
    
    # set the trial seed for use during training
    trial_seed = int(seed)

    env.seed(seed=trial_seed)

    # wrap the env in modified monitor which plots to tensorboard the jumpheight
    env = Monitor(env)
    eval_env = Monitor(eval_env)
    
    # set the callbacks
    evalCallback = EvalCallback(eval_env=eval_env, n_eval_episodes=1, eval_freq=EVALUATION_FREQ, best_model_save_path=models_path, best_model_save_name=f'{data_name}_{int(seed)}', deterministic=True, render=False)

    # create the model
    # open tensorboard with the following bash command: tensorboard --logdir ./logs/

    buffer_size = N_TRAINING_STEPS + 1
    model = TD3("MlpPolicy", env, verbose=1, tensorboard_log=logs_path, buffer_size=buffer_size, learning_starts=ROLLOUT, seed=trial_seed, gamma=GAMMA)

    # train the agent
    
    model.learn(total_timesteps=N_TRAINING_STEPS, callback=evalCallback, tb_log_name=f'{data_name}_{int(seed)}_log')

def runMultiprocessTraining(function, i, n_processors):
    with Pool(processes=n_processors) as pool:
        return pool.map(function, i)

if __name__ == "__main__":
    start = time.time()
    runMultiprocessTraining(trainAgents, TRIAL_SEEDS, 2)
    end = time.time()
    total = end - start
    print(f'Total Time: {total}')