#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:33:56 2019

@author: xiaohezhang
"""
from PIL import Image, ImageFont, ImageDraw
import os
import numpy as np
import pandas as pd
import imgaug as ia
import imgaug.augmenters as iaa
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

breed_name = np.load('./data/breed_name.npy', allow_pickle = True)

def img_aug(ii):
#    ia.seed(42)
    seq = iaa.Sequential([
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ])
    return seq(images=ii)

def resize224(name):
    try:
        img = image.load_img(name, target_size = (224,224))
    except: pass
    return (image.img_to_array(img)/225)

# resize and making prediction for the testing image
# do 100 times augementation on each testing image
def checking(file):
    DIR = './test_pictures/'
    name = DIR+file
    xcheck = np.expand_dims(resize224(name), axis = 0)
    modelbreed = load_model('./model/best_model_v3.h5')
    resultb = {'breed':[],'pred_breed':[]}
    auged_img = []
    for i in range(100):
        auged_img.append(img_aug(xcheck)[0])
    auged_img = np.array(auged_img)
    pred = modelbreed.predict(auged_img)
    for k in range(len(breed_name)):
        for l in range(100):
            total = pred[l][k]
        resultb['pred_breed'].append(total)
        resultb['breed'].append(breed_name[k].split('_')[1])    
    df1= pd.DataFrame(resultb)
    return df1.sort_values('pred_breed', ascending = False).head(5)
   
# draw on the image    
def draw(file):
    DIR = './test_pictures/'
    img = Image.open(DIR+file)
#    img.thumbnail((224,224),Image.ANTIALIAS)
#    img = img.resize((224,224))
    draw = ImageDraw.Draw(img)
    df = checking(file)
    font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 12)
#    background = Image.new('RGB',(140,80) , (255,255,255))
#    img.paste(background)
    for i in range(5):
        draw.text((0,i*15),df.iloc[i,0]+":"+round(df.iloc[i,1]*100,2).astype(str)+'%',(255,0,0),font = font)
    img.show()
    return df

# check similiar images
def similiar(file,x):
    DIR = './downloads/'
    filename = draw(file)
    path = DIR+filename.iloc[x,0]+'/'
    title = f'The #{x+1} prediction of your cat is {filename.iloc[x,0]} with { round(filename.iloc[x,1]*100,3)}% confident and it looks similiar to these'
    print(title)
    fig=plt.figure(figsize=(8, 8))
    for i,n in enumerate(np.random.choice(os.listdir(path),9)):
        im = Image.open(os.path.join(path,n))
        im.thumbnail((224,224),Image.ANTIALIAS)
        im = im.resize((224,224))
        fig.add_subplot(3, 3, i+1)
        plt.imshow(im)
    plt.show()

#similiar('testing.jpeg',0)

