import os
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


def evaluate(model, session, resource_path, classes, post_fix, input_length =100):
    records = {}
    for c in classes:
        #print(f"Loading test data from {os.path.join(resource_path,session,c+post_fix+'.npy')}")
        records[c] = np.load(os.path.join(resource_path,session,c+post_fix+'.npy'),allow_pickle=True)
    
   
    gt = np.arange(len(classes)).repeat(records[c].shape[0])
    preds = []
    for  c in classes:
        sample = records[c].reshape(-1,6*input_length)
        preds.append(model.predict(sample))
    
    preds = np.concatenate(preds, axis = 0)
        
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    print(f'Accuracy : {round(accuracy_score(gt,preds),2)*100}%')
    print(confusion_matrix(gt, preds))
    print()