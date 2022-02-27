import numpy as np
from joblib import load
from scipy import signal
from sklearn.preprocessing import StandardScaler


def DCFilter(data):
    #return data - np.mean(data, axis=0)
    new_data = []
    for d in data:
        new_data.append(d - np.mean(d))
    return np.array(new_data)


def notchFilter(data, f0=60.0, Q=30.0, fs=500):
    b, a = signal.iirnotch(f0, Q, fs)
    data = signal.filtfilt(b, a, data, axis=0)
    return data


def preProcess(data, input_length):
    new_data = []
    for d in data:
        new_data.append(preProcess_1(d, input_length))
    return np.array(new_data)



def preProcess_buffer(data):
    d = data.copy()
    d= DCFilter(d)
    d = np.clip(d, -1000, 1000)
    d /= 1000
    
    # scaler = StandardScaler()
    # scaler.fit(d)
    # d = scaler.transform(d)
    return np.array(d)

def preProcess_1(data, input_length):
    d = data.copy()
    d= DCFilter(d)
    d = signal.resample(d, input_length, axis=-1)
    d = np.clip(d, -1000, 1000)
    d /= 1000
    
    # scaler = StandardScaler()
    # scaler.fit(d)
    # d = scaler.transform(d)
    return np.array(d)
