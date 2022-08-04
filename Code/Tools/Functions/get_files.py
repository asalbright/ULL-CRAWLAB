import os
import glob
from pathlib import Path

def getFiles(file_type, path=None):
    '''
    This function gets all the files of a certain type in a directory.

    Parameters
    ----------
    file_type : str
        The file type to look for
    path : Path
        The path to the directory

    Returns
    -------
    list
        The list of file paths to the files
    path: Path
        The path to the directory if only one file is found
    '''

    if path is None:
        raise ValueError(f'Path not specified for {file_type} file finder.')
    else:
        files = glob.glob(str(path / f'*.{file_type}'))

    if len(files) == 1:
        return files[0]
    elif len(files) == 0:
        raise ValueError(f'No {file_type} files found in {path}.')
    else:
        return files 

def getFileNames(file_type, path=None)->list:
    '''
    This function gets all the file names of a certain type in a directory.

    Parameters
    ----------  
    file_type : str
        The file type to look for
    path : Path
        The path to the directory

    Returns
    -------
    list
        The list of file names to the files
    str
        The name of the file if only one file is found
    '''

    if path is None:
        raise ValueError(f'Path not specified for {file_type} name finder.')
    else:
        files = glob.glob(str(path / f'*.{file_type}'))
        print(f'Number of {file_type} files found: {len(files)}')
        file_names = []
        for f in files:
            file_names.append(os.path.basename(f).split(sep='.')[0])
        
        if len(file_names) == 1:
            return file_names[0]
        elif len(file_names) == 0:
            raise ValueError(f'No {file_type} files found in {path}.')
        else:
            return file_names