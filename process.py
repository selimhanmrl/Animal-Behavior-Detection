from re import A
import deeplabcut
import pandas as pd
import numpy as np
import os
import re
import glob
import matplotlib.pyplot as plt


config_path = r''
output_path = r''

csv_dest_path = ""


csv_path = ''

axis1 = [
    "head_x","head_y",
    "right_ear_x","right_ear_y",
    "left_ear_x","left_ear_y",
    "right_front_leg_x","right_front_leg_y",
    "left_front_leg_x","left_front_leg_y",
    "body_x","body_y",
    "right_back_leg_x","right_back_leg_y",
    "left_back_leg_x","left_back_leg_y",
    "tail_head_x","tail_head_y",
    "tail_x","tail_y",
]

behavior = np.array([
    'standing',
    'walking',
    'grooming'
])

frame_count = 60


def analyze_video(config_path,video_path):
    deeplabcut.analyze_videos(config_path, [str(video_path)], save_as_csv=True, destfolder=output_path)
  

#analyze_video(config_path,video_path)
 


def edit_csv(csv_path, dest_path):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~(df == 'likelihood').any()]
    df.columns = list([df.loc[1]])
    df = df.drop([0,1])
    df = df.drop('coords', axis=1)
    df.columns = axis1
    df.to_csv(dest_path)
    

#edit_csv(csv_path, csv_dest_path)

def visit(path):
    couter = 0
    videos = []
    for root, subdirs ,files in os.walk(path):
        for file in files:
            if file.endswith('.mp4'):
                counter=+1
                videos.append(root + '/'+ file)
                
    #print(videos)
    return videos



def rename(path):
    couter = 0
    videos = []
    for root, subdirs ,files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                new_dest = file

                old_path = os.path.join(root, file)
                new_path = os.path.join(root, new_dest)
                if 'DLC_resnet50_Gtu-MarMay1shuffle1_500000' in file:
                    new_dest = file.replace('DLC_resnet50_Gtu-MarMay1shuffle1_500000','')
                    #new_dest = file.replace('snif','g')
                    new_path = os.path.join(root, new_dest)
                    #os.rename(old_path,new_path)
                if 'snif' in file:
                    new_dest = file.replace('snif','g')
                    #new_dest = file.replace('snif','g')
                    new_path = os.path.join(root, new_dest)
                os.rename(old_path,new_path)
                


def add_label(csv_path, dest_path, label):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~(df == 'likelihood').any()]
    df.columns = list([df.loc[1]])
    df = df.drop([0,1])
    if 'coords' in df.columns:
        df = df.drop('coords', axis=1)
    df.columns = axis1
    df['label'] = np.full(len(df),label)
    df.to_csv(dest_path)
    


def edit(path):

    for root, subdirs ,files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                if '-' not in file:
                    path = os.path.join(root, file)
                    filename, file_extension = os.path.splitext(file)
                    if 's' in filename:
                        add_label(path,path,0)
                    elif 'w' in filename:
                        add_label(path,path,1)
                    elif 'g' in filename:
                        add_label(path,path,2)
            if  file.endswith('.pickle'):
                path = os.path.join(root, file)
                os.remove(path)
            if   file.endswith('.h5'):
                path = os.path.join(root, file)
                os.remove(path)
        
                


def merge_(path):
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
    df_merged  = pd.concat(df_from_each_file, ignore_index=True)
    df_merged = df_merged.loc[:, ~df_merged.columns.str.contains('^Unnamed')]
    print(df_merged.columns)
    df_merged = df_merged.sort_values(by=['label'])

    df_merged.to_csv("merge.csv")



def counts(csv_path):
    df = pd.read_csv(csv_path)
    return len(df)



def create_dataset(path):
    keypoints, labels = [],[]
    count = []
    counter = 0
    for root, subdirs ,files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                path = os.path.join(root, file)
                filename, file_extension = os.path.splitext(file)
                if counts(path) >= 30 :
                    mod = counts(path)%30
                    df = pd.read_csv(path)
                    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                    df[:counts(path)-mod].to_csv(path)

 
                counter +=1


def split_df(path):
    keypoints, labels = [],[]
    count = []
    counter = 0
    name = 0
    for root, subdirs ,files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                path = os.path.join(root, file)
                filename, file_extension = os.path.splitext(file)
                df = pd.read_csv(path)
                arr = np.array_split(df, int(counts(path)/30))
                for i in range(int(counts(path)/30)):
                    df = pd.DataFrame(arr[i])
                    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                    df.to_csv(path.replace(filename,filename+'-'+str(name) ))
                    name += 1
    
                counter +=1



def last_data(path):
    keypoints, labels = [],[]
    count = []
    for root, subdirs ,files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                path = os.path.join(root, file)
                filename, file_extension = os.path.splitext(file)
                df = pd.read_csv(path)
                df = df.loc[:,df.columns != 'label']
            if '-' in filename:
                if 's'  in filename:
                    keypoints.append(df.to_numpy())
                    labels.append(0)
                elif 'w' in filename:
                    keypoints.append(df.to_numpy())
                    labels.append(1)
                elif 'g' in filename:
                     keypoints.append(df.to_numpy())
                     labels.append(2)
            else:
                os.remove(path)

            

            
    return np.array(keypoints),np.array(labels)
                




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

from tensorflow.keras.callbacks import TensorBoard
from  tensorflow.keras.layers import Conv1D, TimeDistributed,Dropout,Flatten
from  tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow import reshape 
from tensorflow.keras.optimizers import SGD

""" log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
 """
def model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,21)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(behavior.shape[0], activation='softmax'))

    return model

def train(model):
    sgd = SGD(learning_rate=0.001)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    history = model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback])

    print(model.summary())


    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['categorical_accuracy'])
    #plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
   # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')




    res = model.predict(X_test)

    print(behavior[np.argmax(res[4])])


    yhat = model.predict(X_test)


    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    multilabel_confusion_matrix(ytrue, yhat)

    print(accuracy_score(ytrue, yhat))

    model.save('v2.h5')

mdl = model()
#train(mdl)

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


mdl.load_weights('v2.h5')
""" 
yhat = mdl.predict(X_test)


ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
print(accuracy_score(ytrue, yhat))
 """

def demo(df,models):
    a = []
    df = df.loc[:,df.columns != 'label']
    a.append(df.to_numpy())
    print(np.array(a).shape)
    pre = models.predict(np.array(a))


    print(behavior[np.argmax(pre[0])])


import time
def demov1(video_path):
    analyze_video(config_path,video_path)
    rename(output_path)
    rename(output_path)
    edit(output_path)
    create_dataset(output_path)
    split_df(output_path)

def pred(mdl):
    keypoints, labels = [],[]
    result = []
    for root, subdirs ,files in os.walk(output_path):
        for file in files:
            if file.endswith('.csv'):
                if '-' in file:    
                    path = os.path.join(root, file)
                    filename, file_extension = os.path.splitext(file)
                    df = pd.read_csv(path)
                    print(path)
                    df = df.loc[:,df.columns != 'label']
                    keypoints.append(df.to_numpy())
                    print(np.array(keypoints).shape)
                    pre = mdl.predict(np.array(keypoints))

                    #result.append(behavior[np.argmax(pre[0])])
                    print(behavior[np.argmax(pre[0])])
            os.remove(os.path.join(root, file))


