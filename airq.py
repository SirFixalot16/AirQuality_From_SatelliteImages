# Imports
import os

import numpy as np
from PIL import Image
import rasterio
import cv2
from matplotlib import pyplot as plt

import pandas
import pathlib



#from berlin import *



# Parametres
path_csv = pathlib.Path(r'C:\Users\Timbo\Documents\Projet\multi\data\airquality.csv')
altitude = 2.4


def preprocess(npdf):
    for i in range(len(npdf)):
        npdf[i][0] = npdf[i][0] + r'.jpg'
        path = pathlib.Path('C:/Users/Timbo/Documents/Projet/multi/data/earth/'+npdf[i][0])
        pathsave = pathlib.Path('C:/Users/Timbo/Documents/Projet/multi/data/earth_downscale/'+npdf[i][0])
        im = Image.open(path)
        im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        im.save(pathsave, "JPEG")

def get_data():
    print('date?')
    dataframe = pandas.read_csv(path_csv, delimiter=',', names=["ct", "airq"])
    #print(dataframe)
    npdf = np.asarray(dataframe)
    print(npdf.shape)
    l = len(npdf)
    x_list, y_list = npdf[:, 0], npdf[:, 1]
    y_list = y_list.astype('float32')
    images = np.zeros(shape=(l, 1024, 1024))
    for i in range(l):
        #im = Image.open('C:/Users/Timbo/Documents/Projet/multi/data/earth_mask/'+x_list[i]+'.jpg')
        im = cv2.imread('C:/Users/Timbo/Documents/Projet/multi/data/earth_mask/'+x_list[i]+'.jpg')
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        images[i] = np.asarray(im).astype('float32')
        images[i] = images[i]/255
        #print(images[i])
        #im = Image.fromarray(images[i].astype('uint8'), 'L')
        #im.show()
    return x_list, y_list, images


def define_model(input_shape, learning_rate):
    import torch.nn as nn
    import torch.optim as optim
    model = nn.Sequential(
        nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=(3,3)
        ),
        nn.ReLU(),
        nn.MaxPool2d((2,2)),
        nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=(3,3)
        ),
        nn.ReLU(),
        nn.MaxPool2d((2,2)),
        nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=(3,3)
        ),
        nn.ReLU(),
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.ReLU(),
    )

    loss_fn = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    from torchsummary import summary
    summary(model, input_shape)
    return model

def train(learning_rate):
    from sklearn.model_selection import train_test_split
    save_at = "macbeth.hdf5"
    save_best2 = ModelCheckpoint (save_at, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='max')
    x, y, images = get_data()
    X_train, X_val, Y_train, Y_val = train_test_split(images[:30,], y[:30,], test_size=0.15, random_state=3)
    input_shape = (1024, 1024, 1)

    model = define_model(input_shape=input_shape, learning_rate=0.001)
    history = model.fit( X_train, Y_train, 
                        epochs = 15, batch_size = 1, 
                        callbacks=[save_best2], verbose=1, 
                        validation_data = (X_val, Y_val))
    
    return history

def acc_plot(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def loss_plot(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def main():
    print('mon mon ya ya')
    his = train(0.001)
    loss_plot(his)

import sys
#np.set_printoptions(threshold=sys.maxsize)

if __name__ == "__main__":
    print('vowlr')
    model = define_model((1, 1024, 1024), 0.0001)
    #main()