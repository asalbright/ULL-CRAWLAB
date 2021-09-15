import gym
import pybullet_envs
# import pybulletgym
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise

# make sure a GPU is available for torch to utilize
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.is_available())

# log paths
save_path = os.getcwd()
logs_path = os.path.join(save_path, 'logs') + '\\'
models_path = os.path.join(save_path, 'models') + '\\'
if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    print('\n' + 'Made Directory: ' + '\n' + logs_path)

if not os.path.exists(models_path):
    os.makedirs(models_path)
    print('\n' + 'Made Directory: ' + '\n' + models_path)

# environment set up
env_id = 'LunarLanderContinuous-v2'
env_train = Monitor(gym.make(env_id))
env_train = DummyVecEnv([lambda: env_train])
env_train = VecNormalize(env_train, norm_obs=True, norm_reward=True)

# model declerathon 
# open tensorboard with the following bash command: tensorboard --logdir ./logs_path/
n_actions = env_train.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = TD3("MlpPolicy", env_train, buffer_size=200000, gamma=0.98, learning_starts=10000,
            gradient_steps=-1, n_episodes_rollout=1, learning_rate=float(1e-3),
            action_noise=action_noise, policy_kwargs=dict(net_arch=[400, 300]), verbose=1,
            tensorboard_log=logs_path)

# variables
n_train_steps = float(25000)

# train the agent
time_start = time.time()
model.learn(total_timesteps=n_train_steps)
time_end = time.time()
print('Time to learn: ', time_end - time_start)
# save the model
model.save(models_path + "\\" + env_id)
env_train.save(models_path+'vec_normalize.pkl')
