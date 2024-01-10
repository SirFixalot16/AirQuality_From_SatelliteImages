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


import sys
#np.set_printoptions(threshold=sys.maxsize)


if __name__ == "__main__":
    print('AirQ')
    #main()