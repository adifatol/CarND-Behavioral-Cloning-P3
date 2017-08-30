# Neural Network Trainer
# nw.py

import numpy as np
import tensorflow as tf
import csv
import cv2
import os.path
import random

EPOCHS = 2

print(' Loading Data ')

#50 percent random
def fifty(percent=50):
    return random.randrange(100) < percent

#load and apply some transformations to images
def loadAndProcess(img_path):
    img       = cv2.imread(img_path)
    hsv       = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert it to hsv
    h, s, v   = cv2.split(hsv)
    v        += random.randrange(10)
    light_hsv = cv2.merge((h, s, v))
    img       = cv2.cvtColor(light_hsv, cv2.COLOR_HSV2RGB)
    return img

with open('new_style/driving_log.csv') as csvfile:
    row_count = sum(1 for row in csvfile)
    print('reading %d lines' % row_count)
    # row_count = 400
    csvfile.seek(0, 0)
    reader    = csv.reader(csvfile)
    angles    = np.zeros(shape=(row_count*3))
    trainimgs = np.zeros(shape=(row_count*3,80,80,3))
    idx       = 0
    failed    = 0
    dropped   = 0
    for line in reader:
        center_img_path = line[0]
        left_img_path   = line[1]
        right_img_path  = line[2]
        center_angle    = float(line[3])
        left_angle      = float(center_angle+0.35)
        right_angle     = float(center_angle-0.35)

        angles[idx]     = center_angle
        angles[idx+1]   = left_angle
        angles[idx+2]   = right_angle

        if (not os.path.exists(center_img_path)):
            print(center_img_path)
            failed+=1
            continue

        chance_to_keep = fifty()
        if ((center_angle < 0.2) and not chance_to_keep):
            dropped+=1
            continue

        imgRGB  = loadAndProcess(center_img_path)
        limgRGB = loadAndProcess(left_img_path)
        rimgRGB = loadAndProcess(right_img_path)

        trainimgs[idx]    = imgRGB
        trainimgs[idx+1]  = limgRGB
        trainimgs[idx+2]  = rimgRGB

        idx+=3
        if (idx == row_count):
            break

print('failed to load %d' % failed)
print('dropped for 0 angle %d' % dropped)
print(angles.shape)
print(trainimgs.shape)

print('print angles plot...')

import scipy.signal as signal

 # checking  angles distribution
import matplotlib
matplotlib.use('agg')
import pylab as plt

fig, ax = plt.subplots( nrows=1, ncols=1 )
ax.plot(angles)
fig.savefig('angles.png')   # save the figure to file

print('Data loaded')

tf.python.control_flow_ops = tf

X = trainimgs
y = angles

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda

from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(80, 80, 3)))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=(80,80, 3)))
model.add(Activation('elu'))
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Activation('elu'))
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(Activation('elu'))
model.add(Dense(512))
model.add(Dropout(.5))
model.add(Activation('elu'))
model.add(Dense(1))

print('Start training')
model.compile(loss='mse', optimizer='adam')
history = model.fit(X, y, nb_epoch=EPOCHS, validation_split=0.2, shuffle=True)

model.save('model.h5')
print('Saved model.h5')

# because some wierd exceptions happens sometimes
# https://stackoverflow.com/questions/40560795/tensorflow-attributeerror-nonetype-object-has-no-attribute-tf-deletestatus
import gc; gc.collect()
