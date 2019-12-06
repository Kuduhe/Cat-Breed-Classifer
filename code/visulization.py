#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 00:56:14 2019

@author: xiaohezhang
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_curve,auc
import imgaug as ia
import imgaug.augmenters as iaa
from tensorflow.keras.preprocessing import image

def visulize():
    DIR = './downloads/'
    
    result = {'breed':[],
              'length':[],
              'height':[],
              'format':[]}
    
    for i in tqdm(os.listdir(DIR)):
        if not i.startswith('.'):
            folder = os.path.join(DIR,i)
            for j in os.listdir(folder):
                path = os.path.join(folder,j)
                filename, file_extension = os.path.splitext(path)
                try:
                    im = Image.open(path)
                    result['breed'].append(i)
                    result['length'].append(im.size[0])
                    result['height'].append(im.size[1])
                    result['format'].append(file_extension)
                except: pass
                
    df = pd.DataFrame(result)
    df.to_csv('./data/visulization.csv')
    return df

#df = visulize()
#df['format'].value_counts()
#
#df = pd.read_csv('./data/visulization.csv').drop('Unnamed: 0', axis = 1)
#plt.figure(figsize = (13,8))
#plt.hist
#plt.hist(df['height'], bins = 20)

#crete ROC score
def ROC():
    predict_xtest = pd.read_csv('./data/predict_xtest.csv')
    y_test = np.load('./data/y_test.npy')
    auc_score = {'breed':[],'score':[]}
    plt.figure(figsize = (15,8))
    for i in range(55):  
        fpr,tpr,_ = roc_curve(y_test[:,i],predict_xtest.iloc[:,i])
        auc_score['score'].append(auc(fpr,tpr))
        auc_score['breed'].append(predict_xtest.columns[i].split('_')[1])
        plt.plot(fpr,tpr)
    baseline = np.linspace(0,1)
    plt.plot(baseline,baseline, color = 'black')
    aucscore = pd.DataFrame(auc_score)
    return aucscore
#ROC().to_csv('./data/aucsocre.csv')

#aucscore = ROC()
#get AUC bar plot
def AUC(aucscore):
    Image.new(size = (224,224), mode = 'RGB', color = (255,255,255))
    ascore.sort_values('score', inplace =True)
    plt.figure(figsize = (8,15))
    plt.barh(ascore['breed'], ascore['score'], color = 'orange')
    plt.title('AUC score for each breed')
    plt.xlabel('AUC score')
    plt.xlim(0.6,1)