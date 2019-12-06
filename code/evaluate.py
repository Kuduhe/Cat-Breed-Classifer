#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:42:06 2019

@author: xiaohezhang
"""

from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


import numpy as np
import pandas as pd
import tensorflow
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.pooling import GlobalAveragePooling2D

from keras.layers.normalization import BatchNormalization

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.optimizers import Adam

X_test = np.load('./data/X_test.npy')
y_test = np.load('./data/y_test.npy')
model = load_model('./model/best_model_v3.h5')
breed_name = np.load('./data/breed_name.npy', allow_pickle = True)


#base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape = (224,224,3))
#x = base_model.output
## x = BatchNormalization()(x)
#x = GlobalAveragePooling2D()(x)
#x = Dense(512, activation='relu')(x)
#x = BatchNormalization()(x)
#x = Activation("relu")(x)
#x = Dropout(0.3)(x)
#x = Dense(256, use_bias=False, kernel_initializer='uniform')(x)
#x = BatchNormalization()(x)
#x = Activation("relu")(x)
#x = Dropout(0.3)(x)
#
#predictions = Dense(55, activation='softmax')(x)
#
#model = Model(input=base_model.input, output=predictions)
#
#for layer in base_model.layers:
#    layer.trainable = False
#
#model.compile(optimizer= Adam(lr=.0001), 
#              loss = 'categorical_crossentropy', 
#              metrics = ['accuracy'])
#
#model.load_weights('./model/best_model_v4.h5')


#df = model.predict(X_test)
#df = pd.DataFrame(df,columns = breed_name)
#df.to_csv('./data/predict_xtest.csv', index = False)

df = pd.read_csv('./data/predict_xtest.csv')

def pred_acc(x):
    count = 0
    for i in range(len(y_test)):
        result = pd.DataFrame()
        result['result'] = df.loc[i,:]
        result['test'] = y_test[i]
        check = result.sort_values('result',ascending = False).head(x)
        count += check['test'].sum()
    return count/len(y_test)

prediction_list = []
for i in tqdm(range(10)):
    prediction_list.append(pred_acc(i+1))


#plt.plot(prediction_list);