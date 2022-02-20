# Task

In this project we classified different face expressions with machine learning tools.
The subject EMG signals has been recorded during the experiment and this signals was processed with Support Vector Machine and Deep Learning (neural network)

# Paradigm
The object of the project was to classify the following 3 different states:

    * Rest    (reference state)
    * Eyebrow (frown)
    * Chew    (jaw compression)  

The EMG signals was recorded with the Mindrove device:

[Official Site](https://mindrove.com/)

[Github Repo](https://github.com/MindRove/SDK_Public)

We recorded 6 channels with 500hz sampling rate.
The electrodes were placed in one line on the subjects forhead.

The experiment contained 5 separate sessions from the same subject, recorded at different times.

During one session the subject executed 20 of each class (60 total) in random order.
Each expression was recorded with the following timeline:

    - 2 second stand by time 
    - The instructions appeared (to get ready)
    - 1 second preparation time
    - Signal appeared to execute the command
    - 2 second record (additional 200 ms added to this part to make sure enough signal is recorded)


# Dataset
The recorded session is available in the resource folder in numpy format. The raw EMG signals has been stored, all filtering and preprocessing applied before the train. Each session is stored in separate folder.
The 3 class has been stored in separated npy files. Each file contains 20 samples. So the files has the following dimension:

(20,6,1000) --> N_sample, N_channel, Time_points(2sec*500hz)

# Training
The project contains 2 separated approach:

    - SVM (Support Vector Machine)
    - Deep learning
These trains are available in the jupyter notebooks.