from scipy import signal
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize


def DCFilter(data):
    new_data = []
    for d in data:
        new_data.append(d - np.mean(d))
    return np.array(new_data)


def notchFilter(data, f0=60.0, Q=30.0, fs=500):
    b, a = signal.iirnotch(f0, Q, fs)
    data = signal.filtfilt(b, a, data, axis=1)
    return np.array(data)


def preProcess(data, input_length):
    data = signal.resample(data, input_length, axis=-1)

    new_data = []
    for d in data:
        d = notchFilter(d)
        scaler = StandardScaler()
        scaler.fit(d)
        d = scaler.transform(d)
        new_data.append(normalize(d, norm="l2"))
    return np.array(new_data)


def preProcess_1(data, input_length):
    d = signal.resample(data, input_length, axis=-1)
    d = notchFilter(d)
    scaler = StandardScaler()
    scaler.fit(d)
    d = scaler.transform(d)
    return np.array(d)
