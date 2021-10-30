#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:29:52 2021

@author: thierryk
"""

import numpy as np
import glob
from PIL import Image



def load_data():

    list_pikachu = glob.glob('data/pikachu/*')
    list_rondoudou = glob.glob('data/rondoudou/*')
    
    HEIGHT = 100
    WIDTH = 100
    CHANNEL = 3
    
    classes = np.array([b'Pikachu', b'Rondoudou'])
    
    # Initialisations
    dataset_arr = np.zeros((len(list_pikachu) + len(list_rondoudou), 
                                    HEIGHT, WIDTH, CHANNEL))
    label = np.zeros((len(list_pikachu) + len(list_rondoudou), 1), dtype='int')
    
    # Generating a Pikachu array type dataset
    for k in range(len(list_pikachu)):
        with Image.open(list_pikachu[k]) as im :
            im = im.resize((HEIGHT, WIDTH), resample=Image.BICUBIC)
            im = im.convert("RGB")
        img_arr = np.array(im)
        dataset_arr[k] = img_arr
        
    # Generating a Rondoudou array type dataset
    i=0
    for k in range(len(list_pikachu), len(dataset_arr)):
        with Image.open(list_rondoudou[i]) as im2 :
            im2 = im2.resize((HEIGHT, WIDTH), resample=Image.BICUBIC)
            im2 = im2.convert("RGB")
        img_arr = np.array(im2)
        dataset_arr[k] = img_arr
        label[k] = 1
        i+=1
    
    n_samples = dataset_arr.shape[0]
    n_val = int(0.2 * n_samples)
    shuffled_indices = np.random.permutation(n_samples)
    train_indices = shuffled_indices[:-n_val] 
    val_indices = shuffled_indices[-n_val:]


    data_train = dataset_arr[train_indices]
    label_train = label[train_indices]
    
    data_val = dataset_arr[val_indices]
    label_val = label[val_indices]
    
    return data_train, label_train, data_val, label_val, classes


if __name__ == 'main':
    data_train, label_train, data_val, label_val, classes = load_data()
    
    INDEX = np.random.randint(data_train.shape[0])
    img_arr = data_train[INDEX]
    img = Image.fromarray(img_arr.astype(np.uint8))
    img.show()
    
    print ("y = " + str(label_train[INDEX]) + ". It's a " + classes[label_train[INDEX].item()].decode("utf-8") +  " picture.")
    