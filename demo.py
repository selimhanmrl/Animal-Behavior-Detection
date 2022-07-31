import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

from tensorflow import keras
import tensorflow.compat.v1 as tf
config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
session = tf.Session(config=config)

tf.keras.backend.set_session(session)

from tkinter import *
from tkinter.filedialog import askopenfile
from tkVideoPlayer import TkinterVideo
import a
import time

config_path = 'Bitirme 2022/GTU-Mar/Gtu-Mar-Selimhan Meral-2022-05-01/config.yaml'
output_path = r'C:\Users\alpag\Desktop\Bitirme 2022\demo'


behavior = np.array([
    'standing',
    'walking',
    'grooming'
])

def model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,21)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(behavior.shape[0], activation='softmax'))

    return model

mdl = model()
mdl.load_weights('v2.h5')

filename = ''
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
                    #print(path)
                    df = df.loc[:,df.columns != 'label']
                    keypoints.append(df.to_numpy())
                    #print(np.array(keypoints).shape)

                    with session.as_default():
                        with session.graph.as_default():
                            pre = mdl.predict(np.array(keypoints))
                    result.append(str(behavior[np.argmax(pre[0])]))
                    
                    print(behavior[np.argmax(pre[0])])
            os.remove(os.path.join(root, file))
    return result

def most_frequent(List):
    return max(set(List), key = List.count)
text = ''

def pred_():
    result = pred(mdl)
    global text
    text = most_frequent(result)
    #var.set(most_frequent(result))
    label.config(text=text)
    label.pack(side = BOTTOM)

def pose():
        global text
        # text  = 'Exracting Keypoints...'
        # label.config(text=text)
        label.config(text='Exracting Keypoints...')
        label.pack(side = BOTTOM)
        root.update()
        a.demov1(file)
        print("\n"+file + ' ===================\n')
        label.config(text='Ready to Predict')
        label.pack(side = BOTTOM)
# result = pred(mdl)
# print(most_frequent(result))


 


import datetime
import tkinter as tk
from tkinter import filedialog
from tkVideoPlayer import TkinterVideo
file = ''
def load_video():
    """ loads the video """
    file_path = filedialog.askopenfilename()

    if file_path:
        vid_player.load(file_path)
        global file
        file = file_path
        play_pause_btn["text"] = "Play"
     


def seek(value):
    """ used to seek a specific timeframe """
    vid_player.seek(int(value))


def play_pause():
    """ pauses and plays """
    if vid_player.is_paused():
        vid_player.play()
        #play_pause_btn["text"] = "Pause"

    else:
        vid_player.pause()
        play_pause_btn["text"] = "Play"


def video_ended(event):
    """ handle video ended """
    play_pause_btn["text"] = "Play"


root = tk.Tk()
root.title("Graduation Project Demo")
root.geometry("700x450")
load_btn = tk.Button(root, text="Load", command=load_video)
load_btn.pack()

play_pause_btn = tk.Button(root, text="Play", command=play_pause)
play_pause_btn.pack()

vid_player = TkinterVideo(scaled=True, master=root)
vid_player.pack(expand=True, fill="both")


skip_plus_5sec = tk.Button(root, text="Pose Estimation", command=lambda: pose())
skip_plus_5sec.pack(side="left")
# var.set(text)
# start_time = tk.Label(root, textvariable=var,relief=RAISED)
# start_time.pack(side="left")
label = Label(root, text=text, relief=RAISED ,font=('Times 14'))


skip_plus_5sec = tk.Button(root, text="Predict", command=lambda: pred_())
skip_plus_5sec.pack(side="right")

root.mainloop()