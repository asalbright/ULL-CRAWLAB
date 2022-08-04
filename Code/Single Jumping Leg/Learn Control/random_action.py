################################################################################################################
# File Name: random_action.py
# Author: Andrew Albright, a.albright1@louisiana.edu
# 
# Description: script for pushing random actions onto into the single jumping let env
# Notes: 
################################################################################################################

import os
import gym
import stable_baselines3
import time
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

from gym_single_leg.gym_single_leg.envs.single_leg_env import SingleLegEnv
from tools.video_functions import VideoWrite


save_dir = Path.cwd()
save_dir = save_dir.joinpath("Temp_Data") 
save_name = "Save_Data"
CONTROL_MODE = {"PC": 0, "VC": 1}

def main():
    ROBOT_TYPE = "USER_SPECIFIED"
    ROBOT_LOCATION = "single_leg_sys/single_leg_sys.urdf"
    EPISODE_STEPS = 240*5
    TOTAL_STEPS = EPISODE_STEPS * 1
    MOTOR_MAX_POS = np.deg2rad(30)
    MOTOR_MAX_VEL = np.deg2rad(330) # 55 RPM -> 330 deg/s
    SPRING_K = 0.75
    SPRING_DAMPING = 1
    FLEX_MAX_POS = np.deg2rad(15)

    # Declare the env
    env = SingleLegEnv(robotType=ROBOT_TYPE,
                       robotLocation=ROBOT_LOCATION,
                       showGUI=True,
                       flexible=True,
                       epSteps=EPISODE_STEPS,
                       maxMotorPos=MOTOR_MAX_POS,
                       maxMotorVel=MOTOR_MAX_VEL,  # RPM
                       maxMotorForce=100,
                       positionGain=SPRING_K,
                       velocityGain=SPRING_DAMPING,
                       maxFlexPosition=FLEX_MAX_POS,
                       captureData=False,
                       saveDataName=None,
                       saveDataLocation=None)

    # video_writer = VideoWrite(fps=240, file_name="Single_Leg").start()

    obs = env.reset()
    reward = []
    contact = []
    time_steps = []

    for ii in range(int(TOTAL_STEPS)):
    # for ii in range(len(MOTOR_INPUTS_0)):
        action = env.action_space.sample()
        action = [0, 0]
        obs, rew, done, _ = env.step(action)
        if env.grounded:
            con = 1
        else:
            con = 0
        reward.append(rew)
        contact.append(con)
        time_steps.append(ii)
        # env.render()
        
        if done:
            obs = env.reset()
            time.sleep(1/240)

    env.close()

# plot the reward vs time
    plt.plot(time_steps, contact)
    plt.plot(time_steps, reward)
    plt.show()

if __name__ == "__main__":
    main()
    