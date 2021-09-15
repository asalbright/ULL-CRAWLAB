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
