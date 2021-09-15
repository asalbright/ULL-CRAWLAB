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

from gym_single_leg.gym_single_leg.envs.single_leg_env import SingleLegEnv


save_dir = Path.cwd()
save_dir = save_dir.joinpath("Temp_Data") 
save_name = "Save_Data"
 
def main():
    ROBOT_TYPE = "USER_SPECIFIED"
    ROBOT_LOCATION = "flexible/basic_flex_jumper/basic_flex_jumper.urdf"
    EPISODE_STEPS = 240*2
    TOTAL_STEPS = EPISODE_STEPS * 3
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
                       controlMode="POSITION_CONTROL",
                       captureData=False,
                       saveDataName=None,
                       saveDataLocation=None)

    obs = env.reset()

    for XX in range(int(TOTAL_STEPS)):
        action = env.action_space.sample()
        obs, _, done, _ = env.step(action)
        time.sleep(1/240)
        joint_states = env.leg.get_joint_states()
        if done:
            obs = env.reset()
            time.sleep(1/30)
    env.close()

if __name__ == "__main__":
    main()
    