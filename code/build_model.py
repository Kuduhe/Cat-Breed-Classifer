#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:07:13 2019

@author: xiaohezhang
"""

import numpy as np
import pandas as pd
import tensorflow
from tensorflow import keras
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.pooling import GlobalAveragePooling2D

from keras.layers.normalization import BatchNormalization

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.optimizers import Adam

X_train = np.load('./data/X_train.npy')
X_test = np.load('./data/X_test.npy')
y_train = np.load('./data/y_train.npy')
y_test = np.load('./data/y_test.npy')

base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape = (224,224,3))
#base_model = InceptionV3(weights='imagenet', include_top=False, input_shape = (224,224,3))

x = base_model.output
# x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dropout(0.3)(x)
x = Dense(256, use_bias=False, kernel_initializer='uniform')(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dropout(0.3)(x)

predictions = Dense(55, activation='softmax')(x)

model = Model(input=base_model.input, output=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer= Adam(lr=.0001), 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

es = EarlyStopping(monitor='val_loss', 
              patience=5,
              verbose=1,
              mode='auto')
mc = ModelCheckpoint('best_model_v4.h5', 
                     monitor='val_acc', 
                     mode='max', 
                     verbose=1, 
                     save_best_only=True)

history = model.fit(X_train, y_train, 
          batch_size = 512, 
          epochs = 20, 
          callbacks = [es,mc], 
          validation_data = [X_test,y_test],
          verbose = 1)


