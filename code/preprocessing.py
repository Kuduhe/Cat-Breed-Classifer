#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:08:32 2019

@author: xiaohezhang
"""

import os
import numpy as np
import pandas as pd
import imgaug.augmenters as iaa
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle


SIZE = 224,224
RGBsize = 224,224,3


def resize224(name):
    try:
        img = image.load_img(name, target_size = SIZE)
    except: pass
    return (image.img_to_array(img)/225)

def img_aug(ii):
    seq = iaa.Sequential([
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ])
    return seq(images=ii)


def fetch_image224():
    DIR = './downloads/'   
    result = {'img':[],
              'label':[]}  
    for folder in tqdm(os.listdir(DIR)):
        if not folder.startswith('.'):
            counter = 0
            for i in os.listdir(DIR+folder):
                name = os.path.join(DIR,folder,i)
                try:
                    if resize224(name).shape == RGBsize:
                        result['img'].append(resize224(name))
                        result['label'].append(f'{folder}')
                        counter += 1
                except: pass
            for j in np.random.choice(os.listdir(DIR+folder), (1000-counter)):
                name1 = os.path.join(DIR,folder,j)
                try:
                    if resize224(name1).shape == RGBsize:
                        result['img'].append(img_aug(np.expand_dims(resize224(name1), axis = 0))[0])
                        result['label'].append(f'{folder}')
                except: pass          
    return np.array(result['img']),np.array(result['label'])


with open('./data/breed_dict.p', 'rb') as fp:
    breed_dict = pickle.load(fp)
    
from collections import defaultdict
re_breed_dict = defaultdict(list)
for key, value in breed_dict.items():
    re_breed_dict[value].append(key)

#Save image and label npy file
#np.save('./data/224x224RGBimgdata.npy',img)
#np.save('./data/224x224RGBlabeldata.npy',label)

X = np.load('./data/224x224RGBimgdata.npy')
label = np.load('./data/224x224RGBlabeldata.npy')

label1 = pd.DataFrame(label)
label1['kind'] = label1[0].map(breed_dict)
 
#One Hot Encode the label data
one = OneHotEncoder(sparse = False)
yy_breed = np.reshape(label, (len(label1),1))
y_breed = one.fit_transform(yy_breed)
breed_name = one.get_feature_names()

#Save label dictionary
#np.save('./data/breed_name.npy',breed_name)

# save training and testing data
#X_train,X_test, y_train,y_test = train_test_split(X,y_breed, test_size = 0.3)
#np.save('./data/X_train.npy',X_train)
#np.save('./data/X_test.npy',X_test)
#np.save('./data/y_train.npy',y_train)
#np.save('./data/y_test.npy',y_test)
