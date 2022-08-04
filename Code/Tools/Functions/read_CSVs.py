import pandas as pd

def readCSVFiles(files)->list:
    '''
    This function reads in a list of txt files and returns a list of dataframes.

    Parameters
    ----------
    files : list
        The list of csv files to read

    Returns
    -------
    list
        The list of dataframes
    '''

    data = []
    try: 
        for f in files:
            df = pd.read_csv(f)
            df.columns = df.columns.str.replace(' ', '')
            df.columns = df.columns.str.replace('#', '')
            data.append(df)
    except:
        df = pd.read_csv(files)
        df.columns = df.columns.str.replace(' ', '')
        df.columns = df.columns.str.replace('#', '')
        data.append(df)

    if len(data) == 1:
        return data[0]
        
    return data