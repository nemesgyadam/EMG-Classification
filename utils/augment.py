import numpy as np
def augment(X, y, ratio = 0.3):
    noise = np.random.normal(0,ratio,X.size)
    noise = np.reshape(noise,X.shape)

    augmented = X + noise
    X_aug = np.concatenate((X,augmented))
    y_aug = np.concatenate((y,y))

 
    return X_aug, y_aug