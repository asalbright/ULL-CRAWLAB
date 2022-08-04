import os
from pathlib import Path
import tkinter
from tkinter import filedialog

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