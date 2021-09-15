# ********************************************
# Author: Andrew Albright
# Date: 03/31/2021

# File containing useful functions

# ********************************************

import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import pandas
import os
import sys

class RewardHeightNotDefined(Exception):
    pass

def getZipFileName(path=None):
    if path is None:
        print('Path not specified for Zip name finder.')
    
    else:
        files = glob.glob(str(path / '*.zip'))
        print(f'Number of Zip files found: {len(files)}')
        file_names = []
        for f in files:
            file_names.append(os.path.basename(f).split(sep='.')[0])
        print(f'File Names found: {file_names}')

    return file_names
def getReward(omega_x, omega_p, x, position_min, position_max, power, power_min, power_max, reward_function=None, power_used=None, height_reached=None, counter=None):
    try:
        if reward_function == "RewardHeightPunishPowerLinear":
            reward = rewardHeightPunishPowerLinear(omega_x, omega_p, x, position_min, position_max, power, power_min, power_max)
            if x > 0 and height_reached[counter - 2] <= 0:
                reward = reward + 25
        elif reward_function == "RewardHeightPunishPowerNonlinearCubed":
            reward = rewardHeightPunishPowerNonlinearCubed(omega_x, omega_p, x, position_min, position_max, power, power_min, power_max)
            if x > 0 and height_reached[counter - 2] <= 0:
                reward = reward + 25
        elif reward_function == "RewardHeightPunishPowerNonlinearCubedSqrt":
            reward = rewardHeightPunishPowerNonlinearCubedSqrt(omega_x, omega_p, x, position_min, position_max, power, power_min, power_max)
            if x > 0 and height_reached[counter - 2] <= 0:
                reward = reward + 25
        elif reward_function == "RewardHeightRewardEfficiency":
            reward = rewardEfficiency(omega_x, omega_p, x, position_min, position_max, power_used, power_min, power_max, counter)
            # if x > 0 and height_reached[counter - 2] <= 0:
            #     reward = reward + 25
        elif reward_function == "RewardHeight": 
            # set max height to achieve as 0.25m so that if we reach that height or higher, we give the agent 
            reward = rewardHeight(x_t=x, x_max=position_max, x_min=position_min)
        else: raise RewardHeightNotDefined

    except RewardHeightNotDefined:
        print("REWARD FUNCTION NOT PROPERLY DEFINED PROPERLY")
        print()
        sys.exit()

    return reward

def rewardHeightPunishPowerLinear(w_x, w_p, x_t, x_min, x_max, p_t, p_min, p_max):
    if x_t > x_max: x_t = x_max
    elif x_t < 0: x_t = 0

    R_x_t = (x_t - x_min) / (x_max - x_min)
    R_p_t = (p_t - p_max) / (p_min - p_max)

    R_t = w_x * R_x_t + w_p * R_p_t
    R_min = 0
    R_max = w_x + w_p

    R_t_norm = (R_t - R_min) / (R_max - R_min)

    return R_t_norm

def rewardHeightPunishPowerNonlinearCubed(w_x, w_p, x_t, x_min, x_max, p_t, p_min, p_max):
    if x_t > x_max: x_t = x_max
    elif x_t < 0: x_t = 0

    R_x_t = (x_t - x_min) / (x_max - x_min)
    R_p_t = (p_t - p_max) / (p_min - p_max)

    R_t = w_x * R_x_t**(3) + w_p * R_p_t
    R_min = 0 
    R_max = w_x + w_p

    R_t_norm = (R_t - R_min) / (R_max - R_min)

    return R_t_norm

def rewardHeightPunishPowerNonlinearCubedSqrt(w_x, w_p, x_t, x_min, x_max, p_t, p_min, p_max):
    if x_t > x_max: x_t = x_max
    elif x_t < 0: x_t = 0

    R_x_t = (x_t - x_min) / (x_max - x_min)
    R_p_t = (p_t - p_max) / (p_min - p_max)

    R_t = w_x * R_x_t**(3) + w_p * R_p_t**(1/2)
    R_min = 0 
    R_max = w_x + w_p

    R_t_norm = (R_t - R_min) / (R_max - R_min)

    return R_t_norm

def rewardHeight(x_t, x_max, x_min):
    if x_t > x_max: x_t = x_max
    elif x_t < 0: x_t = 0

    R_t_norm = (x_t - x_min) / (x_max - x_min)

    return float(R_t_norm)

def rewardEfficiency(w_x, w_p, x_t, x_min, x_max, power_used, p_min, p_max, counter):
    #TODO: ASA - 04/11/2021 - make a reward function that represents rewarding for actual efficiency instead of power use
    if x_t > x_max: x_t = x_max
    elif x_t < 0: x_t = 0

    # R_x_t = (x_t - x_min) / (x_max - x_min)

    e_t = x_t / np.sum(power_used)
    e_max = 0.0025
    e_min = 0
    R_e_t = (e_t - e_min) / (e_max - e_min)

    # R_t = w_x * R_x_t + w_p * R_e_t
    # R_min = 0
    # R_max = w_x + w_p

    # R_t_norm = (R_t - R_min) / (R_max - R_min)

    return float(R_e_t)