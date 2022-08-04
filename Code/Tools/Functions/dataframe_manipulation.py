from pathlib import Path
import pandas as pd

def combineDataInFolder(file_type, path=None) -> pd.DataFrame:
    '''
    This function combines all the data in a folder into one dataframe.

    Parameters
    ----------
    file_type : str
        The file type to look for
    path : Path
        The path to the directory, if None, will ask the user

    Returns
    -------
    pd.DataFrame
        The dataframe containing all the data
    '''

    if path is None:
        path = queueDirectoryPath(Path.cwd(), "Select path where data exists.")

    files = getFiles(file_type, path)
    data = None
    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.replace(' ', '')
        df.columns = df.columns.str.replace('#', '')
        if not data is None:
            data = pd.concat([data, df], axis=1)
        else:
            data = df
    
    return data

def parseDataFrame(df, unique_headers):
    '''
    This function parses a dataframe into a dictionary of dataframes.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to parse
    unique_headers : list
        The list of unique headers to parse the dataframe by

    Returns
    -------
    dict
        The dictionary of dataframes
    '''

    # Make sure all columns are strings without following characters
    df.columns = df.columns.str.replace(' ', '')
    df.columns = df.columns.str.replace('#', '')

    data_frames = {}
    for header in unique_headers:
        temp_df = None
        for col in df.columns:
            if header in col:
                if not temp_df is None:
                    temp_df = pd.concat([temp_df, df[col]], axis=1)
                else:
                    temp_df = df[col]
        # append to dict of dataframes
        data_frames[header] = temp_df
    
    return data_frames

def dfAverageStd(df):
    '''
    This function calculates the average and standard deviation of a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to calculate the average and standard deviation of

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        The dataframes containing the average and standard deviation
    '''
    
    return df.mean(axis=1), df.std(axis=1)