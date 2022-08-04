###############################################################################
# evaluation.py
#
# A script for evaluating the model produced by taining.py and saving
# the data 
#
# Copied: 02/15/2022 from Pogo Stick Work https://github.com/CRAWlab/CRAWLAB-Student-Code---2020/tree/master/Andrew%20Albright/Code/Pogo%20Stick
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
# Modified:
# * 
###############################################################################

import gym
import numpy as np
import pandas as pd 
import os
import sys
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import datetime


from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from tools.functions import *
from tools.video_functions import *

from gym_single_leg.gym_single_leg.envs import SingleLegEnv

PLOT_DATA = False
SAVE_FIG = False
SHOW_FIG = False

# Define the max number of env steps
ROBOT_TYPE = "USER_SPECIFIED"
ROBOT_LOCATION = "single_leg_sys/single_leg_sys.urdf"
EP_STEPS = 240*2
MOTOR_MAX_POS = np.deg2rad(30)
MOTOR_MAX_VEL = np.deg2rad(330) # 55 RPM -> 330 deg/s
SPRING_K = 0.75
SPRING_DAMPING = 0.75


def evaluate_agents(agents, agents_path, save_path):

    # Loop through all the agents and test them
    for agent in agents:

        # Define the agents jump type and reward function
        env = SingleLegEnv(robotType=ROBOT_TYPE,
                           robotLocation=ROBOT_LOCATION,
                           showGUI=True,
                           flexible=True,
                           epSteps=EP_STEPS,
                           maxMotorVel=MOTOR_MAX_VEL,
                           positionGain=SPRING_K,
                           velocityGain=SPRING_DAMPING,
                           captureData=True,
                           saveDataName=agent,
                           saveDataLocation=save_path)
        # Wrap the env
        env = Monitor(env)

        # load the agent from the models path
        model_id = agent
        model = TD3.load(path=agents_path / model_id)

        # Create a video Writer
        file_name = save_path / f"{model_id}_test.mp4"
        video_writer = VideoWrite(file_name=file_name).start()

        # Evaluate the agent
        obs = env.reset()

        done, state = False, None
        while not done:
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = env.step(action)
            # env.render()

            # Update the video writer frame
            video_writer.frame = env.get_camera_image()
            # video_writer._write_frame()

        env.close()
        video_writer.stop()

    # Combine the data
    data = combineDataInFolder(file_type="csv", path=save_path)
    # Save the data
    path = save_path / "Combined_Data"
    if not os.path.exists(path):
        os.makedirs(path)
    save_name = "single_leg"           # 1 and 2 parts of the file names
    data.to_csv(path / f"{save_name}_Combined.csv")
    
    if PLOT_DATA:
        if SAVE_FIG:
            save_path = save_path.parent / "figures"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        plot_data(path, save_path)
    

def plot_data(data_path, save_path):
    files = getFiles("csv", data_path)
    data = readCSVFiles(files)
    unique_headers = ['Time', 'Reward', 'Height', 'Motor1Pos', 'Motor1Vel', 'Motor2Pos', 'Motor2Vel']
    data = parseDataFrame(data[0], unique_headers)

    for header in unique_headers:

        X_MEAN, X_STD = dfAverageStd(data["Time"])
        Y_MEAN, Y_STD = dfAverageStd(data[header])

        # Set the plot size - 3x2 aspect ratio is best
        fig = plt.figure(figsize=(9,6))
        ax = plt.gca()

        # Define the X and Y axis labels
        plt.xlabel(r'Time (s)', fontsize=22, weight='bold', labelpad=5)
        plt.ylabel(header, fontsize=22, weight='bold', labelpad=10)

        plt.plot(X_MEAN, Y_MEAN, linewidth=2, linestyle='-', label=header)
        plt.fill_between(X_MEAN, Y_MEAN - Y_STD, Y_MEAN + Y_STD, alpha=0.2)
                
        # uncomment below and set limits if needed
        # plt.xlim(0,1.25)
        # plt.ylim(bottom=None, top=1.75)

        # Create the legend, then fix the fontsize
        # leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
        # ltext  = leg.get_texts()
        # plt.setp(ltext, fontsize=18)

        # Adjust the page layout filling the page using the new tight_layout command
        plt.tight_layout(pad=0.5)

        # save the figure as a high-res pdf in the current folder
        filename = f"{header}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = save_path / filename
        if SAVE_FIG is True:
            plt.savefig(path, transparent=True)

    if SHOW_FIG:
        plt.show()
    else:
        plt.close('all')

if __name__ == "__main__":
    # Query the user for the path to the models
    models_path = getDirectoryPath(Path.cwd(), header="Select models directory.")
    # create a directory to save the data in the data's parent directory
    save_path = models_path.parent / "figures_data"
    # if the path exists check check if the user wants to overwrite it
    if os.path.exists(save_path): 
        answer = guiYesNo("Overwrite Existing Data?", f"Data exits in {save_path}, do you want to overwrite it?")
        if answer == 'yes': # delete the folder
            shutil.rmtree(save_path)
        if answer == 'no': # exit the program
            sys.exit()
    # if the path does not exist create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    agents = getFileNames(file_type="zip", path=models_path)
    evaluate_agents(agents, models_path, save_path)