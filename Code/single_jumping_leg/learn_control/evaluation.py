################################################################################################################
# File Name: evaluation.py
# Author: Andrew Albright, a.albright1@louisiana.edu
# 
# Description: Script for evaluating the agents that are trained
# Notes: 
################################################################################################################

import gym
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from tools.functions import getZipFileName

from gym_single_leg.gym_single_leg.envs.single_leg_env import SingleLegEnv

# Declare folders for logging data
def getNameofFile():
    '''
    Prompts user to enter the path of a file where the training data exists. 
    Creates a \plotting_data folder within path to save data.

    Within entered path there should exist: 

    \models 
    \logs  
    README.md 

    If the files are named wrong there will be an error thrown.
    '''
    ROOT = tk.Tk()
    ROOT.withdraw()
    title = "File Location"
    prompt = "Enter the path to the training data: "
    # Get path from user
    path = simpledialog.askstring(title, prompt)
    # If user closes program
    if path is None:
        raise Exception("File path is does not exit or user closed program.")
        sys.exit()
    # If user enters non-existant path
    save_path = Path(path)
    if not os.path.exists(save_path):
        raise Exception("File path is does not exit or user closed program.")
        sys.exit()
    # Set the paths according to how they should be saved
    models_path = save_path.joinpath("models")
    logs_path = save_path.joinpath("logs")
    save_path = save_path.joinpath("plotting_data")
    # Make the save path if it does not exits
    if not os.path.exists(save_path): os.makedirs(save_path)
    # If the models path does not exist or is named wrong
    if not os.path.exists(models_path):
        raise Exception("Models path does not exist or name of models file is not 'models'.")
        sys.exit()
    # Print what the models path defined and the save path defined
    print(f"\nThe models will be queued from:\n {models_path}.\n")
    print(f"The plotting data will be saved to:\n {save_path}.")

    return save_path, models_path

def evaluateAgents(save_path, models_path):
    # Call function to get agent names from models_path folder
    agent_names = getZipFileName(path=models_path)

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

    for agent in agent_names:
        # make gym environment
        save_name = agent.split('_')
        save_name = f"{save_name[1]}_{save_name[2]}"
        
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
                           captureData=True,
                           saveDataName=save_name,
                           saveDataLocation=save_path)        
        

        # wrap the env in a Monitor and DummyVecEnv wrapper
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])

        # load the agent from the models path
        model_id = agent

        model = TD3.load(path=models_path / model_id)

        # Evaluate the agent
        done, state = False, False
        obs = env.reset()
        while not done:
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = env.step(action)
        env.close()

if __name__ == "__main__":
    save_path, models_path = getNameofFile()
    evaluateAgents(save_path, models_path)