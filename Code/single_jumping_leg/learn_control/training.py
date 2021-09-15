################################################################################################################
# File name: training.py
# Author: Andrew Albright, a.albright1@louisiana.edu
#
# Description: script for training an agent to jump the single_leg to jump high
# Notes:
# TODO: June 18, 2021, ASA, figure out the eval call back line 81
################################################################################################################

import os
from pathlib import Path
import time
import numpy as np 
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass
import gym
import torch
import stable_baselines3
from stable_baselines3 import TD3
from tools.custom_callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from gym_single_leg.gym_single_leg.envs.single_leg_env import SingleLegEnv

# Check if a GPU is available for torch to utilize
if(torch.cuda.is_available()):
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
else: print('No GPU available.')

# Define the environment ID's and set Environment parameters
ROBOT_TYPE = "USER_SPECIFIED"
ROBOT_LOCATION = "flexible/basic_flex_jumper/basic_flex_jumper.urdf"
EPISODE_STEPS = 240*2
MOTOR_MAX_POS = np.deg2rad(30)
MOTOR_MAX_VEL = np.deg2rad(330) # 55 RPM -> 330 deg/s
FLEX_MAX_POS = np.deg2rad(15)
SPRING_K = 0.75
SPRING_DAMPING = 1
ENV_ID = "SingleLeg-v0"

# Set up the Training parameters
NUM_TRIALS = 6
TOTAL_EPISODES = 10
TOTAL_STEPS = EPISODE_STEPS * TOTAL_EPISODES
ROLLOUT = EPISODE_STEPS * 10
GAMMA = 0.99

# Set up the training seeds
INITIAL_SEED = 70504
EVALUATION_FREQ = EPISODE_STEPS * 5
SAVE_FREQ = 100000
RNG = np.random.default_rng(INITIAL_SEED)
TRIAL_SEEDS = RNG.integers(low=0, high=1000, size=(NUM_TRIALS))

# Set up the training save paths
data_name = f'{ENV_ID}'
save_path = Path.cwd()
save_path = save_path.joinpath(f'trained_{data_name}')
logs_path = save_path.joinpath('logs')
models_path = save_path.joinpath('models')
if not os.path.exists(logs_path):
    os.makedirs(logs_path)
if not os.path.exists(models_path):
    os.makedirs(models_path)

# Function for training agents
def train_agent(seed):
    # Define model name
    model_name = f'model_{int(seed)}'

    # Set up training env
    env = SingleLegEnv(robotType=ROBOT_TYPE,
                       robotLocation=ROBOT_LOCATION,
                       showGUI=False,
                       flexible=True,
                       epSteps=EPISODE_STEPS,
                       maxMotorPos=MOTOR_MAX_POS,
                       maxMotorVel=MOTOR_MAX_VEL,  # RPM
                       maxMotorForce=100,
                       positionGain=SPRING_K,
                       velocityGain=SPRING_DAMPING,
                       maxFlexPosition=FLEX_MAX_POS,
                       controlMode="POSITION_CONTROL",
                       rewardCase="HEIGHT",
                       captureData=False,
                       saveDataName=None,
                       saveDataLocation=None)

    # Set up evaluation env (used to generate "Best Agents")
    eval_env = SingleLegEnv(robotType=ROBOT_TYPE,
                       robotLocation=ROBOT_LOCATION,
                       showGUI=False,
                       flexible=True,
                       epSteps=EPISODE_STEPS,
                       maxMotorPos=MOTOR_MAX_POS,
                       maxMotorVel=MOTOR_MAX_VEL,  # RPM
                       maxMotorForce=100,
                       positionGain=SPRING_K,
                       velocityGain=SPRING_DAMPING,
                       maxFlexPosition=FLEX_MAX_POS,
                       controlMode="POSITION_CONTROL",
                       rewardCase="HEIGHT",
                       captureData=False,
                       saveDataName=None,
                       saveDataLocation=None)
    
    # Set the trial seed for use during training
    trial_seed = int(seed)
    env.seed(seed=trial_seed)

    # Wrap the training env in monitor which plots to tensorboard
    env = Monitor(env)
    eval_env = Monitor(eval_env)
    
    # Set the callbacks
    # Evaluation callback gets called every EVAL_FREQ and prints the reward
    # FIXME: June 18, 2021, ASA, there is a problem with this having to do with having multiple p.connects I think. We get through the eval, but on return we lose something having to do with the training env. For now we are not using this
    eval_callback = EvalCallback(eval_env=eval_env, 
                                n_eval_episodes=1, 
                                eval_freq=EVALUATION_FREQ, 
                                best_model_save_name=model_name,
                                best_model_save_path=models_path,
                                deterministic=True, 
                                render=False)

    # Callback to replicate the evalCallback. Would rather use that so that we know we are saving only "better" models, however this will save a model every 50k steps as is.
    save_callback = CheckpointCallback(SAVE_FREQ, models_path, name_prefix=model_name)

    # Create the model
    # Open tensorboard with the following bash command: tensorboard --logdir ./logs/
    buffer_size = TOTAL_STEPS + 1
    model = TD3("MlpPolicy", env, verbose=1, 
                                  tensorboard_log=logs_path, 
                                  buffer_size=buffer_size, 
                                  learning_starts=ROLLOUT, 
                                  seed=trial_seed, 
                                  gamma=GAMMA)

    # Train the agent and log the data to tensorboard
    model.learn(total_timesteps=TOTAL_STEPS, 
                callback=eval_callback, 
                tb_log_name=f'{int(seed)}_log')

    # Close the environment
    env.close()

def multiprocess_training(function, i, n_processors):
    with Pool(processes=n_processors) as pool:
        return pool.map(function, i)

if __name__ == "__main__":
    # start = time.time()
    # multiprocess_training(train_agent, TRIAL_SEEDS, 2)
    # end = time.time()
    # total = end - start
    # print(f'Total Time: {total}')
    train_agent(12345)
