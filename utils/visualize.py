from matplotlib import pyplot as plt
import numpy as np

def showMe2(data, r=None, std_threshold=200):

    plt.rcParams["figure.figsize"] = [17, 2]
    fig, (c1, c2, c3, c4, c5, c6) = plt.subplots(1, 6)
    if r is not None:
        c1.set_ylim(r[0], r[1])
        c2.set_ylim(r[0], r[1])
        c3.set_ylim(r[0], r[1])
        c4.set_ylim(r[0], r[1])
        c5.set_ylim(r[0], r[1])
        c6.set_ylim(r[0], r[1])
    
    c1.plot(data[0])
    c2.plot(data[1])
    c3.plot(data[2])
    c4.plot(data[3])
    c5.plot(data[4])
    c6.plot(data[5])
    
    middle = 0
    for i in range(100):
        std_sum = 0
        for d in data:
            std_sum+=np.std(d[:i*10+10])
        std_avg = std_sum/len(data)
        if std_avg>std_threshold:
            middle = i*10+10
            break

    #FUCK ME
    if middle<250:
        middle = 250
    if middle>len(data[0])-250:
        middle = len(data[0])-250

    c1.axvline(x=middle-250, color = 'r')
    c1.axvline(x=middle+250, color = 'g')

    plt.show()



def showMe(data, r=[-1,1]):
    plt.rcParams["figure.figsize"] = [5, 5]
    fig, ax = plt.subplots(facecolor ='#00AAAA')
    ax.set_ylim(r[0], r[1])

    for d in data:
        ax.plot(d)

    

    # start, end = getRange(data, std_threshold)
    # ax.axvline(x=start, color = 'r')
    # ax.axvline(x=end+250, color = 'g')
    plt.show()
    

def showHistory(history):
    plt.rcParams["figure.figsize"] = [5, 5]
    for key in history.history.keys():

        if "val_" not in key and "lr" != key:
            try:
                plt.clf()
                plt.plot(history.history[key])
                plt.plot(history.history["val_" + key])
                plt.ylabel(key)
                plt.xlabel("epoch")
                plt.legend(["train", "validation"], loc="upper left")
                plt.show()
            except:
                ...


def visualize(data, cla):
    range = [-400,400]
    fig = plt.gcf()
    fig.canvas.manager.window.raise_()
    mngr = plt.get_current_fig_manager()
    
    min_mean, max_mean, std_mean = stat(data)
 
    plt.cla()
    plt.ylim(range[0], range[1])
    plt.text(200, range[1]*1.1, f'{cla}', fontsize=10)
    plt.text(500, range[1]*1.1, f'Min: {min_mean}', fontsize=10)
    plt.text(700, range[1]*1.1, f'Max: {max_mean}', fontsize=10)
    plt.text(900, range[1]*1.1, f'Std: {std_mean}', fontsize=10)    
    

    for i, c in enumerate(data):
        # c-=60
        # c+=i*20
        plt.plot(c, label='eeg')
    

    mngr.window.setGeometry(1150,100,800, 800)
    fig.canvas.flush_events()
    plt.draw()
    # plt.pause(0.001)
    #plt.cla()
    plt.show()

def VisulaizeCell(dd, std_threshold):
    """
    Draw all channels in one figure
    """
    range = [int(std_threshold*-2), int(std_threshold*2)]
    data = dd.copy()
    plt.clf()

    #############################################################
    min_mean, max_mean, std_mean = stat(data)
    plt.text(60, range[1]*1.1, f'|Min: {min_mean}', fontsize=10)
    plt.text(80, range[1]*1.1, f'|Max: {max_mean}', fontsize=10)
    plt.text(100, range[1]*1.1, f'|Std: {std_mean}', fontsize=10)
    #############################################################
    plt.ylim(range)
    

    for i, c in enumerate(data):
        # c-=60
        # c+=i*20
        plt.plot(c, label='command')
    
    plt.show()
    


def Vis(dd,  range = [-1,1]):
    data = dd.copy()

   

    fig = plt.gcf()
    fig.canvas.manager.window.raise_()
    mngr = plt.get_current_fig_manager()
    
    min_mean, max_mean, std_mean = stat(data)

    plt.subplot(1, 2, 1)
    plt.text(0, range[1]*1.1, f'Min: {min_mean}', fontsize=10)
    plt.text(20, range[1]*1.1, f'Max: {max_mean}', fontsize=10)
    plt.text(40, range[1]*1.1, f'Std: {std_mean}', fontsize=10)
   
    plt.ylim(range[0], range[1])
    

    for i, c in enumerate(data):
        # c-=60
        # c+=i*20
        plt.plot(c, label='eeg')
    

    mngr.window.setGeometry(1150,100,800, 800)

   
    plt.draw()
    plt.pause(0.001)
    plt.cla()

    return min_mean, max_mean, std_mean