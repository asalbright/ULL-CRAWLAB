##########################################################################################
# MX28_Plotting.py
#
# Script to plot data for single leg, two dynamixel MX-28 servo end effector along an array-based trajectory
# Based on MX28Controller.py made by Eve Dang
#
# dynamixel_sdk module from: https://github.com/ROBOTIS-GIT/DynamixelSDK/tree/master/python/src/dynamixel_sdk
#
# MX-28 Info: https://emanual.robotis.com/docs/en/dxl/mx/mx-28/
#
# Created: 8/31/2021 Eve Dang	
#
##########################################################################################

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import csv
import glob
import datetime
import pandas
from Decimal_data_converter import negate, binaryToDecimal, DecimalToVelocity



save_path = Path.cwd()
Jumping_Data_path = save_path / 'Experimental Platform'
Jumping_Data_path = Jumping_Data_path / 'Jumping_Data'
Data_Figures_path = save_path / 'Data_Figures'
if not os.path.exists(Data_Figures_path):
        os.makedirs(Data_Figures_path)
print(Jumping_Data_path)
print(save_path)

files = glob.glob(str(Jumping_Data_path / '*.csv'))
print(f'Number of Files found: {len(files)}')
data = {}                                                     # dict to store agent data
file_names = []                                               # list to store file names for naming agents in dict
for f in files:
    file_name = os.path.basename(f).split(sep='_',)            # get file name and split it up by '_'
    # uncomment print statment to determine what numbers are needed in file_name
    print(file_name)
    file_name = file_name[2].split(sep='.')    # choose the parts of file name you want to keep using the previous print statment
    print(file_name)
    file_names.append(file_name[0])                              # add the agents name to the list of file names
    df = pandas.read_csv(f)                               # read in the first .csv file
    df_columns = list(df)                                     # get the list of data column names
    for column in df_columns:                                                               
        df_columns[df_columns.index(column)] = column.replace(' ', '').replace('#', '')     # get rid of spaces and #'s in the column names
    df.columns = df_columns                                   # set the .csv file column names to the cleaned up names
    data[f] = df.to_dict('series')                    # turn the .csv into a dict and place it in the data dict
    for column in data[f]:
            data[f][column] = data[f][column].to_numpy()                    # convert the columns in the agent dict in the data dict to numpy arrays

print(f'Experiment Key Names: {list(data)}\n')                                                    # print out the agents names for using later in plotting
print(f"""Experiment's Data Key Names: {list(data[next(iter(data))])}\n""")    


for experiment in data:
    for index in range(len((data[experiment]['Servo1Speeds']))):
        data[experiment]['Servo1Positions'][index] = int(data[experiment]['Servo1Positions'][index]) * 0.29     # deg
        data[experiment]['Servo2Positions'][index] = int(data[experiment]['Servo2Positions'][index]) * 0.29
        data[experiment]['Servo1Speeds'][index] = int(data[experiment]['Servo1Speeds'][index])*0.111
        data[experiment]['Servo2Speeds'][index] = int(data[experiment]['Servo2Speeds'][index])*0.111


for experiment in data:
    # Set the plot size - 3x2 aspect ratio is best
    fig = plt.figure(figsize=(6,4))
    ax = plt.gca()
    plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

    # Change the axis units font
    plt.setp(ax.get_ymajorticklabels(),fontsize=18)
    plt.setp(ax.get_xmajorticklabels(),fontsize=18)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Turn on the plot grid and set appropriate linestyle and color
    ax.grid(True,linestyle=':', color='0.75')
    ax.set_axisbelow(True)

    # Define the X and Y axis labels
    plt.xlabel('Time (s)', fontsize=22, weight='bold', labelpad=5)
    plt.ylabel('Positions', fontsize=22, weight='bold', labelpad=10)
 
    plt.plot(data[experiment]['Time(s)'], data[experiment]['Servo1Positions'], linewidth=2, linestyle='-',label=r'Servo1')
    plt.plot(data[experiment]['Time(s)'], data[experiment]['Servo2Positions'], linewidth=2, linestyle='--',label=r'Servo2')

    # uncomment below and set limits if needed
    # plt.xlim(0,5)
    # plt.ylim(0,15)  # Plot the full range

    # Create the legend, then fix the fontsize
    leg = plt.legend(loc='upper right', ncol = 2, fancybox=True)
    ltext  = leg.get_texts()
    plt.setp(ltext,fontsize=18)

    # Adjust the page layout filling the page using the new tight_layout command
    plt.tight_layout(pad=0.5)

    angle_figure_filename = f'Figure_{file_names[files.index(experiment)]}_Position.png'
    path = Data_Figures_path / angle_figure_filename 
    plt.savefig(path)

    # show the figure
    plt.show()
