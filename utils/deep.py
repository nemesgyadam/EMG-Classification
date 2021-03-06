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


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as layers
from tensorflow.keras.layers import GlobalMaxPooling2D, Activation, Dense, Conv1D, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, GlobalMaxPooling1D
from tensorflow.keras import optimizers

from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


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
        acc, c_m = evaluate_session(model, session, classes, post_fix,input_length=input_length,  log = log)
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


def m2tex(model):
    stringlist = []
    model.summary(line_length=70, print_fn=lambda x: stringlist.append(x))
    del stringlist[1:-4:2]
    del stringlist[-1]
    for ix in range(1,len(stringlist)-3):
        tmp = stringlist[ix]
        stringlist[ix] = tmp[0:31]+"& "+tmp[31:59]+"& "+tmp[59:]+"\\\\ \hline"
    stringlist[0] = "Model: test \\\\ \hline"
    stringlist[1] = stringlist[1]+" \hline"
    stringlist[-4] = stringlist[-4]+" \hline"
    stringlist[-3] = stringlist[-3]+" \\\\"
    stringlist[-2] = stringlist[-2]+" \\\\"
    stringlist[-1] = stringlist[-1]+" \\\\ \hline"
    prefix = ["\\begin{table}[]", "\\begin{tabular}{lll}"]
    suffix = ["\end{tabular}", "\caption{Model summary for test.}", "\label{tab:model-summary}" , "\end{table}"]
    stringlist = prefix + stringlist + suffix 
    out_str = " \n".join(stringlist)
    out_str = out_str.replace("_", "\_")
    out_str = out_str.replace("#", "\#")
    print(out_str)

def get_model(input_shape, n_classes):
    inspected_chanels= input_shape[0]
    input_length=     input_shape[1]
    l2 = 0.000001

    input_layer = keras.Input(shape = (inspected_chanels,input_length,1), name='input')

    x     = layers.AveragePooling2D(pool_size=(1,5))(input_layer) # resample


   
   
    x     = layers.Conv2D(256, kernel_size=(1,5), padding='same', activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
    x     = layers.BatchNormalization()(x)
    x     = layers.AveragePooling2D(pool_size=(1,5))(x)

    x     = layers.Conv2D(64, kernel_size=(4,1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
    x     = layers.BatchNormalization()(x)
    x     = layers.AveragePooling2D(pool_size=(4,1))(x)

    x     = layers.Dense(100,kernel_regularizer=regularizers.l2(l2))(x)
    x     = layers.Flatten()(x)

 
   
    x     = layers.Dense(20,kernel_regularizer=regularizers.l2(l2))(x)
    x     = layers.BatchNormalization()(x)
    x     = layers.Dropout(.1)(x)

    


    output = layers.Dense(n_classes, activation='softmax')(x)
    model = keras.Model(inputs=input_layer, outputs=output)


    return model