import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt


from joblib import load
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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

def cut_out(data,
        input_length = 500,
        resample_rate = 10,
        early_focus_ratio = 0.01,
        percentage_before_magic = 0.2 
    ):
        

        resample_to = int(data.shape[-1]/resample_rate)
        border = int(resample_to/10)
        maxs = []
        for c in data:
            gradient = np.gradient(c)
            gradient = signal.resample(gradient,resample_to)

            # Prefer early peaks
            adjust = np.linspace(0,early_focus_ratio,resample_to)
            gradient = gradient-adjust

            # Dont watch sides
            gradient[:border] = 0
            gradient[-border:] = 0

            magic = (np.argmax(gradient)*resample_rate)+resample_rate/2
            maxs.append(magic)
        magic = np.array(maxs).mean()
    
        
        start = magic - input_length*percentage_before_magic
        end = magic + input_length*(1-percentage_before_magic)

        if start < 0:
            start = 0
            end = input_length
        if end > data.shape[-1]:
            end = data.shape[-1]
            start = data.shape[-1] - input_length


        return data[:,int(start):int(end)]

def create_labels(X):
    y = []
    for i, r in enumerate(X):
        if X[r].shape[0] == 0:
            l = np.array([0])
        else:
            l = np.ones(X[r].shape[0])*i
        y = y + l.tolist()
    y = np.array(y)
    return y


def evaluate_session(model, session, classes, post_fix, input_length =100, log = False):
    records = {}
    for c in classes:
        #print(f"Loading test data from {os.path.join(resource_path,session,c+post_fix+'.npy')}")
        records[c] = np.load(os.path.join(session,c+post_fix+'.npy'),allow_pickle=True)
    
    # for r in records:
    #     records[r] = records[r][:,:,1000:]

    gt = create_labels(records)
    preds = []
    for  c in classes:
        #sample = records[c].reshape(-1,6*input_length)

        #sample = np.array([cut_out(s) for s in records[c]])
        if records[c].shape[0] == 0:
            #print('No data for class {}'.format(c))
            ######   ITT VAGYOK
            preds.append([0])
        else:
            sample = records[c].reshape(-1,records[c].shape[-2]*records[c].shape[-1])
            #input(model.predict(sample))
            preds.append(model.predict(sample))
    
    preds = np.concatenate(preds, axis = 0)
        
   
    accuracy = round(accuracy_score(gt,preds),2)
    cm = confusion_matrix(gt,preds)
    if log:
        print(f'Accuracy : {accuracy*100}%')
        print(confusion_matrix(gt, preds))
        print()
    return int(accuracy*100), cm

def evaluate_set(model, set, classes, post_fix, input_length = 100, log = False):
    results  = pd.DataFrame(columns=["Subject", "Session", "Accuracy"])
    confusion_matrixes = {}
    
    for session in tqdm(set):
        full_session_name =session
        #print("Evaluating session: {}".format(session))
        acc, c_m = evaluate_session(model, session, classes, post_fix, log=log)
        session = session.replace('\\','/')
        subject = session.split('/')[-2]
        session = session.split('/')[-1]

        results = results.append({
        "Subject": subject,
        "Session":  session,
        "Accuracy": acc
        }, ignore_index=True)
        confusion_matrixes[full_session_name] = c_m
        
    results = results.astype({"Subject": str, "Session": str, "Accuracy": int})   
    if log:
        print(f'Global accuracy: {round(results.mean().to_numpy()[0],2)}%')
        by_subject = results.groupby('Subject').mean()
        print(by_subject)
    return results, confusion_matrixes



###################################
#######      NEW      #############
###################################



def remove2channel(data):
    
    mins = []
    for i, channel in enumerate(data):
        mins.append(np.min(channel))
    data = np.delete(data, np.argmin(mins), 0)

    maxs = []
    for i, channel in enumerate(data):
        maxs.append(np.max(channel))
    data = np.delete(data, np.argmax(maxs), 0)
   
    return data


def clip(data, clip_value=2000):
    data = np.clip(data, -clip_value, clip_value)
    data /= clip_value
    return data

def show_me_cut(source, results, magic, r=[-1,1]):
    plt.rcParams["figure.figsize"] = [16, 4]
    fig, ax = plt.subplots(facecolor ='#A0A0A0')

    plt.subplot(1, 4, 1)
    for d in source:
        plt.plot(d)
        plt.ylim(r[0], r[1])
        plt.axvline(x=magic)

    
    for i, result in enumerate(results):
        plt.subplot(1, 4, i+2)
        for d in result:
            plt.plot(d)
            plt.ylim(r[0], r[1])
    plt.show()

def cut_out(
    data,
    draw=True,
    input_length=500,
    resample_rate=10,
    early_focus_ratio=0.01,
    percentage_before_magic=0.4,
    augmentation_shift=50,
):

    resample_to = int(data.shape[-1] / resample_rate)
    border = int(resample_to / 10)
    maxs = []
    for c in data:
        # Calculate gradient
        gradient = np.gradient(c)
        gradient = signal.resample(gradient, resample_to)
        gradient = np.abs(gradient)
        # Prefer early peaks
        adjust = np.linspace(0, early_focus_ratio, resample_to)
        gradient = gradient - adjust

        # Dont watch sides
        gradient[:border] = 0
        gradient[-border:] = 0
 
        highest_gradient = (np.argmax(gradient) * resample_rate) + resample_rate / 2
        maxs.append(highest_gradient)

    magic = np.array(maxs).mean()

    start = magic - input_length * percentage_before_magic
    end = start+input_length

    

    try:
        a = data[:, int(start - augmentation_shift) : int(end - augmentation_shift)]
        if a.shape[-1]!=500:
            a = data[:, :input_length]
    except:
        a = data[:, :input_length]
        print(a.shape)
        
    try:
        b = data[:, int(start) : int(end)]
        if b.shape[-1]!=500:
            b = data[:, int(input_length/2):int(input_length*1.5)]
    except:
        b = data[:, int(input_length/2):int(input_length*1.5)]
        print(b.shape)
       
    try:
        c = data[:, int(start + augmentation_shift) : int(end + augmentation_shift)]
        if c.shape[-1]!=500:
            c = data[:, -input_length:]
    except:
        c = data[:, -input_length:]
        print(c.shape)

    try:
        concat = np.array([a, b, c])
    except:
        print("WTF")
        print(a.shape)
        print(b.shape) 
        print(c.shape)
        input()
    if draw:
        show_me_cut(data, concat, magic)
    return concat
