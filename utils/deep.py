import pandas as pd
import numpy as np
import os
from tqdm.notebook import tqdm


from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def smoothLabels(label, factor = 0.05):
    label *= (1 - factor)
    label += (factor / len(label))
    return label

def oneHot(label, num_classes):
    label = to_categorical(label,num_classes=num_classes)
    return smoothLabels(label)

def applyOneHot(data, n_classes):
    new = []
    for y in data:
        new.append(oneHot(y,num_classes=n_classes))
    return np.array(new)



def DCFilter(data):
    new_data = []
    for d in data:
        new_data.append(d - np.mean(d))
    return np.array(new_data)


def notchFilter(data, f0=60.0, Q=30.0, fs=500):
    b, a = signal.iirnotch(f0, Q, fs)
    data = signal.filtfilt(b, a, data, axis=1)
    return np.array(data)


def preProcess(d):
    #d = notchFilter(d,100)

    scaler = StandardScaler()
    scaler.fit(d)
    d = scaler.transform(d)

    #d = normalize(d, norm ='l2')
    return d
    

def preProcess_1(data, input_length):
    d = signal.resample(data, input_length, axis=-1)
    d = notchFilter(d)
    scaler = StandardScaler()
    scaler.fit(d)
    d = scaler.transform(d)
    return np.array(d)

def create_labels(X):
    y = []
    for i, r in enumerate(X):
        l = np.ones(X[r].shape[0])*i
        y = y + l.tolist()
    y = np.array(y)
    return y

def evaluate_session(model, session, classes, post_fix, input_length =100, log = False):
    records = {}
    for c in classes:
        #print(f"Loading test data from {os.path.join(resource_path,session,c+post_fix+'.npy')}")
        records[c] = np.load(os.path.join(session,c+post_fix+'.npy'),allow_pickle=True)
    
    
    gt = create_labels(records)
    #gt = applyOneHot(gt, len(classes))
   
    
    preds = []
    for c in classes:
       
        X = records[c]
        if X.shape[0] == 0:
            #print('No data for class {}'.format(c))
            preds.append([])
        else:
            X = X.reshape((-1, 4, 500))

            pred = np.argmax(model.predict(X), axis=1)
        
            
            preds.append(pred)
    
    preds = np.concatenate(preds, axis = 0)
        
 
    accuracy = round(accuracy_score(gt,preds),2)

    if log:
        print(f'Accuracy : {accuracy*100}%')
        print(confusion_matrix(gt, preds))
        print()
    return int(accuracy*100)


def evaluate_set(model, set, classes, post_fix, input_length = 100, log = False):
    results  = pd.DataFrame(columns=["Subject", "Session", "Accuracy"])
    for session in tqdm(set):
        #print("Evaluating session: {}".format(session))
        acc = evaluate_session(model, session, classes, post_fix,input_length=input_length,  log = log)
        session = session.replace('\\','/')
        subject = session.split('/')[-2]
        session = session.split('/')[-1]
        results = results.append({
        "Subject": subject,
        "Session":  session,
        "Accuracy": acc
        }, ignore_index=True)
        
    results = results.astype({"Subject": str, "Session": str, "Accuracy": int})   
    print(f'Global accuracy: {round(results.mean().to_numpy()[0],2)}%')
    by_subject = results.groupby('Subject').mean()
    print(by_subject)
    #return resul