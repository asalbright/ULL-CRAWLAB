###############################################################################
# evaluation.py
#
# A script for evaluating the model produced by the RL algorithm
#
# Created: 11/30/2021
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
# Modified:
#
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


from stable_baselines3 import PPO, TD3
from stable_baselines3.common.monitor import Monitor
from functions import getFiles, getFileNames, readCSVFiles, parseDataFrame, queueDirectoryPath, combineDataInFolder, dfAverageStd, guiYesNo

from gym_three_link_robot.gym_three_link_robot.envs import ThreeLinkRobotArmEnv

CAPTURE_DATA = True
PLOT_DATA = True
SAVE_FIG = True
ENV_ID = "three_link_robot_arm-v0"
EP_STEPS = 300
RANDOM_TARGET = False

PARAMS = [9.86006961, 10.17448818,  9.96544221,  2.31096247,  1.8923805 ,
        1.79665703]

def evaluate_agents(agents, models_path, save_path, targets=[(3, 3)]):
    # Loop through all the agents and test them
    
    for agent in agents:
        power_used = 0.0
        targets_reached = 0
        targets = targets = [(4, 4), (3, 3)]

        for target in targets:
            # Create and wrap the environment
            env = gym.make(ENV_ID)
            env.capture_data = CAPTURE_DATA
            env.data_location = save_path
            env.specified_pos = target
            env.ep_steps = 65

            # load the agent from the models path
            model_id = agent

            model = TD3.load(path=models_path / model_id)

            # Evaluate the agent
            done, state = False, None
            obs = env.reset()
            data = None


            modify_env_design(env, PARAMS)

            while not done:
                action, state = model.predict(obs, state=state, deterministic=True)
                obs, reward, done, info = env.step(action)
                if info["target_reached"]:
                    targets_reached += 1
                env.render()
            # Print the power used
            # print(f"Episode Power Used: {env.env.robot.power_used}")
            power_used += env.robot.power_used
            env.close()

        print(f"Total Power Used: {power_used}")
        print(f"Targets Reached: {targets_reached}")

    # Combine the data
    data = combineDataInFolder(file_type="csv", path=save_path)
    # Save the data
    path = save_path / "Combined_Data"
    if not os.path.exists(path):
        os.makedirs(path)
    save_name = 'Random_Action'           # 1 and 2 parts of the file names
    data.to_csv(path / f"{save_name}_Combined.csv")
    
    if PLOT_DATA:
        plot_combined_data(data, save_path=save_path)

    return power_used

def plot_combined_data(data, save_path):
    unique_headers = ['Time', 'Reward', 'XPos', 'YPos', 'Joint1Pos', 'Joint1Vel', 'Joint2Pos', 'Joint2Vel', 'Joint3Pos', 'Joint3Vel']
    data = parseDataFrame(data, unique_headers)

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
        filename = f"{header}_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')}.svg"
        path = save_path / filename
        if SAVE_FIG is True:
            plt.savefig(path, transparent=True)

    plt.show()

def genetate_targets(range=[1, 4.2]):
    xy = np.linspace(range[0], range[1], num=3)
    targets = [(xy[0],xy[0]), (xy[0],xy[1]), (xy[0],xy[2]), 
               (xy[1],xy[0]), (xy[1],xy[1]), (xy[1],xy[2]), 
               (xy[2],xy[0]), (xy[2],xy[1]), (xy[2],xy[2])]
    return targets

def modify_env_design(env, params):
        # Modify the environment parameters
        env.robot.modify_design(params)

        return env

if __name__ == "__main__":
    # Query the user for the path to the models
    models_path = queueDirectoryPath(Path.cwd(), header="Select models directory.")
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