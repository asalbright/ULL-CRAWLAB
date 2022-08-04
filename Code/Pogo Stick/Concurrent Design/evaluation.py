###############################################################################
# evaluation.py
#
# A script for evaluating the model produced by pogo_stick_training and saving
# the data 
#
# Created: 02/23/2021
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
# Modified:
# * 10/02/2021 - Updated saving data so it is all saved in one file
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
from functions import *

from gym_pogo_stick.gym_pogo_stick.envs import PogoStickControlEnv

REWARD_FUNCTIONS = ["Height", "Efficiency", "SpecHei", "SpHeEf"]
JUMP_TYPES = {"One": 1, "Stutter": 2}
PLOT_DATA = True
SAVE_FIG = True
SHOW_FIG = False

# Define the max number of env steps
ENV_TYPE = "Nonlinear"
EP_STEPS = 400

def evaluate_models(models:list, params:list, models_path, save_path):

    # Loop through all the models and test them with their respective parameters
    for ii in range(len(models)):
        agent = models[ii]
        agent_params = params[ii]

        # Define the models jump type and reward function
        for jump_type in JUMP_TYPES:
            if jump_type in agent:
                NUM_JUMPS = JUMP_TYPES[jump_type]
        for reward_function in REWARD_FUNCTIONS:
            if reward_function[0:5] in agent:
                REWARD_FUNCTION = reward_function

        # Set the env
        env = PogoStickControlEnv(numJumps=NUM_JUMPS,
                                linear=ENV_TYPE,
                                epSteps=EP_STEPS, 
                                evaluating=True,
                                rewardType=REWARD_FUNCTION,
                                captureData=True, 
                                saveDataName=agent,
                                saveDataLocation=save_path)
        # Wrap the env
        env = Monitor(env)

        # load the agent from the models path
        model_id = agent
        model = TD3.load(path=models_path / model_id)

        # Evaluate the agent
        obs = env.reset()
        
        if not agent_params is None:
            env.pogo_stick.modify_design(params=agent_params)

        done, state = False, None
        while not done:
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = env.step(action)
            # env.render()
        env.close()

    # Combine the data
    data = combineDataInFolder(file_type="csv", path=save_path, axis=1)
    # Save the data
    path = save_path / "Combined_Data"
    if not os.path.exists(path):
        os.makedirs(path)
    save_name = models[0].split("_")[1] + models[0].split("_")[2]           # 1 and 2 parts of the file names
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
    unique_headers = ['Time', 'Reward', 'Input', 'RodPos', 'RodVel', 'ActPos', 'ActVel']
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

def get_designs(dpath=None):
    fpath = getFilePath(dpath, header="Select Designs File")

    designs = readCSVFile(fpath)
    params = []
    # Get the column names
    cols = designs.columns
    dict_keys = ["k", "zeta"]
    # Get the number of designs
    num_designs = len(designs[cols[0]])
    # Convert the data to a list of dictionaries with keys 
    for ii in range(num_designs):
        params.append(dict(zip(dict_keys, designs.iloc[ii])))

    return params

if __name__ == "__main__":
    # Query the user for the path to the models
    model_paths = getDirectoryPaths(Path.cwd(), header="Select models directory.")
    for models_path in model_paths:
        # create a directory to save the data in the data's parent directory
        save_path = models_path.parents[1] / "figures_data"
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

        # Get the file paths of all the models
        models = getFileNames(file_type="zip", path=models_path)
        designs = get_designs(models_path.parents[0])

        # Get the designs associated with the models
        evaluate_models(models, designs, models_path, save_path)
    
    