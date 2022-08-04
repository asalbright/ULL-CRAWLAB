###############################################################################
# convert_to_csv.py
#
# A script for converting tensorboard log files to .csv files
# 
# Notes: https://stackoverflow.com/questions/42355122/can-i-export-a-tensorflow-summary-to-csv
#
# Copied: 09/21/2021
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
#
###############################################################################


import os
import numpy as np
import pandas as pd
from pathlib import Path
import tkinter 
from tkinter import filedialog

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def getFilePath(current_path, header=None):
    '''
    This function queues a directory path for the user to select.

    Parameters
    ----------
    current_path : Path
        The current path to the directory
    header : str
        The header of the gui

    Returns
    -------
    Path
        The path to the directory
    '''
    if header is None:
        header = "Select the Directory Where Data is Stored"

    root = tkinter.Tk()
    root.withdraw() #use to hide tkinter window
    currdir = current_path
    tempdir = filedialog.askdirectory(parent=root, initialdir=currdir, title=header)
    if not os.path.exists(currdir):
        raise ValueError("\nPath does not exits!")
    elif not os.path.isdir(tempdir):
        raise ValueError("\nNot a directory!")
    elif tempdir is None:
        raise ValueError("\nNo directory selected!")
    if len(tempdir) > 0:
        print("You chose %s" % tempdir)

    return Path(tempdir)

def tabulate_events(dpath, spath):

    final_out = {}
    for dname in os.listdir(dpath):
        print(f"Converting run {dname}",end="")
        ea = EventAccumulator(os.path.join(dpath, dname)).Reload()
        tags = ea.Tags()['scalars']

        out = {}

        for tag in tags:
            tag_values=[]
            wall_time=[]
            steps=[]

            for event in ea.Scalars(tag):
                tag_values.append(event.value)
                wall_time.append(event.wall_time)
                steps.append(event.step)

            out[tag]=pd.DataFrame(data=dict(zip(steps,np.array([tag_values,wall_time]).transpose())), columns=steps,index=['value','wall_time'])

        if len(tags)>0:      
            df= pd.concat(out.values(),keys=out.keys())
            path = spath / f'{dname}.csv'
            df.to_csv(path)
            print("- Done")
        else:
            print('- Not scalers to write')

        final_out[dname] = df


    return final_out

if __name__ == '__main__':
    '''
    Enter the path to the logs folder below, and this will create a folder, 'training_data'
    where .csv files will be saved for all the logs data along with a final concatinated
    'all_results.csv' file.
    '''

    logs_path = getFilePath(current_path=Path.cwd(), header="Select the Directory Where Logs is Stored")
    save_path = Path(logs_path).parents[0] / 'training_data'
    if not os.path.exists(save_path): os.makedirs(save_path)

    steps = tabulate_events(logs_path, save_path)
    path = save_path / 'all_result.csv'
    pd.concat(steps.values(),keys=steps.keys()).to_csv(path)