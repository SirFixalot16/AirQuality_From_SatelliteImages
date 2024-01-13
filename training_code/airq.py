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

def greyscale_to_transparent(img:Image)->Image:
    data = img.getdata()
    
    newData = []
    for item in data:
        if item[0] > 100:
            newData.append((255, 255))
        else:
            newData.append((0, 0))
    #newData = np.array(newData)
    #print(newData)
    #im = Image.fromarray(newData)
    img.putdata(newData)
    return img

def white_to_color(img:Image, color, pixel_shape=3000, background=False)->Image:
    red, green, blue = 0, 0, 0
    if (color == "red"): red = 255
    elif (color == "green"): green = 255
    elif (color == "blue"): blue = 255

    res = Image.new('RGBA', (pixel_shape, pixel_shape))
    data = img.getdata()
    newData = []

    for item in data:
        if item[0] == 255 and item[1] == 255:
            newData.append((red, green, blue, 255))
        elif background==True:
            newData.append((0, 0, 0, 255))
        else:
            newData.append((0, 0, 0, 0))
    
    res.putdata(newData)
    return res

def merge_masks(r, g:Image, b)->Image:
    # Open as PIL greyscale
    r = Image.open(r)
    r = r.convert('LA')
    #g = Image.open(g)
    #g = g.convert('LA')
    b = Image.open(b)
    b = b.convert('LA')

    # Convert to numpy array
    npr = np.array(r)
    #npg = np.array(g)
    npb = np.array(b)
    npr[npr <= 100] = 0
    #npg[npg <= 100] = 0
    npb[npb <= 100] = 0

    # Convert to channels
    r = Image.fromarray(npr, 'LA')
    #g = Image.fromarray(npg, 'LA')
    b = Image.fromarray(npb, 'LA')
    r = greyscale_to_transparent(r)
    #g = greyscale_to_transparent(g)
    b = greyscale_to_transparent(b)
    r = white_to_color(img=r, color='red', background=True)
    #g = white_to_color(img=b, color='green')
    b = white_to_color(img=b, color='blue')
    r.paste(g, g)
    r.paste(b, b)
    print(r)

    return r

def generate_greens(img)->Image:
    im = Image.open(img)
    im.thumbnail((3000, 3000), Image.Resampling.LANCZOS)
    HSVim = im.convert('HSV')
    #im.show()
    im_og = np.array(im)
    im_arr = np.array(HSVim)
    H = im_arr[:,:,0]
    lo, hi = 100, 140
    lo = int((lo * 255) / 360)
    hi = int((hi * 255) / 360)
    green = np.where((H>lo) & (H<hi))
    im_og[green] = [0,0,0]
    im=Image.fromarray(im_og)
    data = im.getdata()
    newData = []
    for item in data:
        if item[0]==0 and item[1]==0 and item[2]==0:
            newData.append((0, 255, 0, 255))
        else:
            newData.append((0, 0, 0, 0))
        #print(item)
    res = Image.new('RGBA', (3000, 3000))
    res.putdata(newData)
    
    return res

import sys
#np.set_printoptions(threshold=sys.maxsize)

def main():
    b = Image.open('./data/dog/b_hanoi_caugiay.jpg')
    b = b.convert('LA')
    r = Image.open('./data/dog/r_hanoi_caugiay.jpg')
    r = r.convert('LA')
    npb = np.array(b)
    npr = np.array(r)
    npb[npb <= 100] = 0
    npr[npr <= 100] = 0
    p = Image.fromarray(npb, 'LA')
    p = greyscale_to_transparent(p)
    p = white_to_color(img=p, color='blue')
    #p.show()
    e = Image.fromarray(npr, 'LA')
    e = greyscale_to_transparent(e)
    e = white_to_color(img=e, color='red', background=True)
    #e.show()
    #m = merge_images_as_channels(p, e, 'blue', 'red')
    e.paste(p, p)
    print(e)
    e.show()


if __name__ == "__main__":
    print('vowlr')

    #link = './data/test/hanoi_caugiay.jpg'
    #im = generate_greens(link)
    #im.show()
    #im.save('./data/ex.png')
    #data = im.getdata()
    #newData = []
    #for item in data:
    #    if item[1] < 100 and item[0] > 50 and item[2]>50:
    #        newData.append((0, 255, 0, 255))
    #    else:
    #        newData.append((0, 0, 0, 0))
    #    #print(item)
    #im.putdata(newData)
    #im.show()
    #p.save('./data/test.png', 'PNG')
    #print(npb)

    images = r'C:\Users\Timbo\Documents\Projet\multi\data\test_images/'
    roads = r'C:\Users\Timbo\Documents\Projet\multi\data\test_roads/'
    buildings = r'C:\Users\Timbo\Documents\Projet\multi\data\test_buildings/'
    pathsave = r'C:\Users\Timbo\Documents\Projet\multi\data\test_masks/'
    green = os.listdir(images)
    
    for item in green:
        im = merge_masks(
            r = pathlib.Path(roads+item),
            g = generate_greens(images+item),
            b = pathlib.Path(buildings+item)
        )
        im = im.convert('RGB')
        im.save(pathsave+item[:-4]+'.png', 'PNG')