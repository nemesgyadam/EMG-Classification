import numpy as np
from scipy import signal

def stat(data):
    """
    Count the std, max and min of each channel
    And returns avg of them
    """
    mins = []
    maxs = []
    stds = []
    for channel in data:
        mins.append(np.min(channel))
        maxs.append(np.max(channel))
        stds.append(np.std(channel))
    min_mean = round(np.mean(mins),0)
    max_mean = round(np.mean(maxs),0)
    std_mean = round(np.mean(stds),0)
    return min_mean, max_mean, std_mean


def getRange(data,std_threshold, signal_length = 500):
    """
    Get the range of the data
    based on the std theshold
    """
    half = int(signal_length/2)
    additional_edge4aug = int(std_threshold/10)
    middle = 0
    for i in range(100):
        std_sum = 0
        for d in data:
            std_sum+=np.std(d[:i*10+10])
        std_avg = std_sum/len(data)
        if std_avg>std_threshold:
            middle = i*10+10
            break

    ## Handle edge cases
    if middle<half+additional_edge4aug:
        middle = half+additional_edge4aug
        
    if middle>len(data[0])-(half+additional_edge4aug):
        middle = len(data[0])-(half+additional_edge4aug)


    return middle-half,middle+half

def preProcess(data, clip_value):
    """
    Clip and resample the data
    """

    data = signal.redata(data, 100, axis = -1)

    data = np.clip(data, -clip_value, clip_value) 
    data /= clip_value
    return data