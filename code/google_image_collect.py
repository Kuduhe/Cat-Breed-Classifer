#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 21:34:58 2019

@author: xiaohezhang
"""
import requests
from bs4 import BeautifulSoup
from google_images_download import google_images_download

#getting the breed list
url = 'https://tica.org/breeds/browse-all-breeds'
soup = BeautifulSoup(requests.get(url).content,'lxml')
table = soup.find_all('a',{'data-parent':'#set-rl_sliders-1'})
breed_list = []
for i in table:
    breed_list.append(i.text[3:])
#this is the list of all breed
breed_list                            

def download_image(x):    
    down = google_images_download.googleimagesdownload()
    arguments = {"keywords":x,
                 "limit":99,
                 'related_images':True,
                 'no_directory':True
                 }
    return down.download(arguments) 

download_image('Birman cat')      

