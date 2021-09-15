import gym
import pybullet_envs
# import pybulletgym
import numpy as np
import matplotlib.pyplot as plt
import os

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# log paths
save_path = os.getcwd()
logs_path = os.path.join(save_path, 'logs') + '\\'
models_path = os.path.join(save_path, 'models') + '\\'
if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    print('Made Directory: ', logs_path)
import os
if not os.path.exists(models_path):
    os.makedirs(models_path)
    print('Made Directory: ', models_path)

# environment set up
env_id = 'LunarLanderContinuous-v2'
env = Monitor(gym.make(env_id))
env = DummyVecEnv([lambda: env])
env = VecNormalize.load(models_path+'vec_normalize.pkl', env)
model = TD3.load(path=models_path+env_id)

# evaluate the agent

# env.training = False
# env.norm_reward = False

env.render()
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, render=True)
print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Deploy the agent

# env.render()
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
# env.close()