def get_avg_dict(list):
    '''
    This function calculates the average value of each key from a list of dictionaries.

    Parameters
    ----------
    dict : list
        The list if dictonaries to calculate the average value of each key

    Returns
    -------
    dict
        The dictionary with the average value of each key
    '''
    
    avg_dict = {}
    for d in list:
        for key in d.keys():
            if key in avg_dict.keys():
                avg_dict[key].append(d[key])
            else:
                avg_dict[key] = [d[key]]
    
    for key in avg_dict.keys():
        avg_dict[key] = np.mean(avg_dict[key])

    return avg_dict