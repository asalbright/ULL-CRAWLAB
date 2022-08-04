###############################################################################
# random_action.py
#
# A script for evaluating performance with a random actions
#
# Created: 11/29/2021
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
#
###############################################################################

import os
from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt
import datetime
import gym

from stable_baselines3.common.monitor import Monitor
from functions import getFiles, getFileNames, readCSVFiles, parseDataFrame, queueDirectoryPath, combineDataInFolder, dfAverageStd, guiYesNo

from gym_three_link_robot.gym_three_link_robot.envs import ThreeLinkRobotArmEnv

CAPTURE_DATA = True
RANDOM_TARGET = True
PLOT_DATA = True
SAVE_FIG = False

ENV_ID = "three_link_robot_arm-v0"
EP_STEPS = 300
EPISODES = 1

def random_action(save_path, targets=(4.2, 4.2)):
    for target in targets:
        # Create and wrap the environment
        env = gym.make(ENV_ID)
        env.capture_data = CAPTURE_DATA
        env.random_target = RANDOM_TARGET
        env.data_location = save_path
        env.specified_pos = target

        # Logs will be saved in log_dir/monitor.csv
        env = Monitor(env)  
        obs = env.reset()

        for _ in range(EP_STEPS * EPISODES):
            action = env.action_space.sample()
            # print(action)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                print(f"Power Used: {env.env.robot.power_used}")
                env.reset()
                env.close()

    # Combine the data
    data = combineDataInFolder(file_type="csv", path=save_path)
    # Save the data
    path = save_path / "Combined_Data"
    if not os.path.exists(path):
        os.makedirs(path)
    save_name = 'Random_Action'           # 1 and 2 parts of the file names
    data.to_csv(path / f"{save_name}_Combined.csv")
    
    if PLOT_DATA:
        plot_combined_data(data)

def plot_combined_data(data):
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
        # filename = 'MostEfficientAgent_{}.png'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
        # path = save_path / filename
        # if SAVE_FIG is True:
        #     plt.savefig(path, transparent=True)

    plt.show()

def genetate_targets(range=[1, 4.2]):
    xy = np.linspace(range[0], range[1], num=3)
    targets = [(xy[0],xy[0]), (xy[0],xy[1]), (xy[0],xy[2]), 
               (xy[1],xy[0]), (xy[1],xy[1]), (xy[1],xy[2]), 
               (xy[2],xy[0]), (xy[2],xy[1]), (xy[2],xy[2])]
    return targets

if __name__ == "__main__":
    # create path to save data
    save_path = Path.cwd() / "Random_Action_Results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    random_action(save_path=save_path, targets=genetate_targets())