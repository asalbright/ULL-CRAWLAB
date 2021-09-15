###############################################################################
# pogo_stick_training.py
#
# A script for training the pogo_sick env
#
# Created: 02/23/2021
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
import gym
import torch
import stable_baselines3
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise
from custom_callbacks import EvalCallback
import pogo_stick_jumping


# log paths
save_path = Path.cwd()
save_path = save_path.joinpath(f'zdata_test_spring_k')
logs_path = save_path.joinpath(f'logs_test_spring_k')
models_path = save_path.joinpath(f'models_test_spring_k')

if not os.path.exists(logs_path):
    os.makedirs(logs_path)
if not os.path.exists(models_path):
    os.makedirs(models_path)

# make sure a GPU is available for torch to utilize
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(torch.cuda.current_device()))
# print(torch.cuda.is_available())

# wrap the env in modified monitor which plots to tensorboard the jumpheight

# training variables
NUM_TRIALS = 1
N_TRAINING_STEPS = 10000
ROLLOUT_STEPS = 10000
INITIAL_SEED = 70504
RNG = np.random.default_rng(INITIAL_SEED)
TRIAL_SEEDS = RNG.integers(low=0, high=INITIAL_SEED, size=(NUM_TRIALS))

# create spring constants to test
nominal_k = 97280
variance = 0.75
high_k = nominal_k + nominal_k * variance
low_k = nominal_k - nominal_k * variance

SPRING_CONSTANTS = np.arange(int(low_k), int(high_k)+1, int(variance*nominal_k/1))

for trial in range(NUM_TRIALS):
    
    for spring_k in SPRING_CONSTANTS:

        env_id = 'pogo-stick-jumping-v01'
        env = gym.make(env_id)
        env.MAX_STEPS = 200
        env.ONE_JUMP = True
        env.omega_x = 0.95

        eval_env = gym.make(env_id)
        eval_env.MAX_STEPS = 200
        eval_env.ONE_JUMP = True
        eval_env.EVALUATING = True
        
        # set the reward function of the env
        env.REWARD_FUNCTION = "RewardHeight"
        eval_env.REWARD_FUNCTION = "RewardHeight"
        
        # set the spring k to evaluate with
        env.SPRING_K = spring_k
        eval_env.SPRING_K = spring_k

        # set trial seeds
        trial_seed = int(TRIAL_SEEDS[trial])
        env.seed(seed=trial_seed)
        eval_env.seed(seed=trial_seed)
        
        # create the action noise to be used in training
        # TODO: 03/17/2021 - ASA - Not currently used
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        
        # wrap the eval env in a monitor wrapper
        eval_env = Monitor(eval_env)
        
        # set the callbacks
        save_frequency = 100000
        eval_frequency = 10000
        # callback for saving the model every save_frequency timesteps
        saveCallback = CheckpointCallback(save_freq=save_frequency, save_path=models_path, name_prefix=env_id)
        # callback for chaining all callbacks
        evalCallback = EvalCallback(eval_env=eval_env, n_eval_episodes=1, eval_freq=eval_frequency, best_model_save_path=models_path, best_model_save_name=f'model_spring_{int(spring_k)}', deterministic=True, render=False)
        callbacks = CallbackList([saveCallback])

        # create the model
        # open tensorboard with the following bash command: tensorboard --logdir ./logs/
        buffer_size = N_TRAINING_STEPS + 1
        model = TD3("MlpPolicy", env, verbose=0, tensorboard_log=logs_path, buffer_size=buffer_size, learning_starts=ROLLOUT_STEPS, seed=trial_seed)

        # wrap the env in a monitor wrapper
        env = Monitor(env)
        
        # train the agent
        model.learn(total_timesteps=N_TRAINING_STEPS, callback=evalCallback, tb_log_name=f'log_spring_{int(spring_k)}')